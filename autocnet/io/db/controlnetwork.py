from csv import (writer as csv_writer, QUOTE_MINIMAL)
from io import StringIO

import pandas as pd
import numpy as np
import shapely.wkb as swkb
from plio.io import io_controlnetwork as cnet
from autocnet.io.db.model import Measures


def db_to_df(engine, sql = """
SELECT measures."pointid",
        points."pointType",
        points."apriori",
        points."adjusted",
        points."pointIgnore",
        points."referenceIndex",
        points."identifier",
        measures."id",
        measures."serialnumber",
        measures."sample",
        measures."line",
        measures."measureType",
        measures."imageid",
        measures."measureIgnore",
        measures."measureJigsawRejected",
        measures."aprioriline",
        measures."apriorisample"
FROM measures
INNER JOIN points ON measures."pointid" = points."id"
WHERE
    points."pointIgnore" = False AND
    measures."measureIgnore" = FALSE AND
    measures."measureJigsawRejected" = FALSE AND
    measures."imageid" NOT IN
        (SELECT measures."imageid"
        FROM measures
        INNER JOIN points ON measures."pointid" = points."id"
        WHERE measures."measureIgnore" = False and measures."measureJigsawRejected" = False AND points."pointIgnore" = False
        GROUP BY measures."imageid"
        HAVING COUNT(DISTINCT measures."pointid")  < 3)
ORDER BY measures."pointid", measures."id";
"""):
        """
        Given a set of points/measures in an autocnet database, generate an ISIS
        compliant control network.
        Parameters
        ----------
        path : str
               The full path to the output network.
        flistpath : str
                    (Optional) the path to the output filelist. By default
                    the outout filelist path is genrated programatically
                    as the provided path with the extension replaced with .lis.
                    For example, out.net would have an associated out.lis file.
        sql : str
              The sql query to execute in the database.
        """
        df = pd.read_sql(sql, engine)

        # measures.id DB column was read in to ensure the proper ordering of DF
        # so the correct measure is written as reference
        del df['id']
        df.rename(columns = {'pointid': 'id',
                             'pointType': 'pointtype',
                             'measureType': 'measuretype'}, inplace=True)
        df['id'] = df.apply(lambda row: f"{row['identifier']}_{row['id']}", axis=1)

        #create columns in the dataframe; zeros ensure plio (/protobuf) will
        #ignore unless populated with alternate values
        df['aprioriX'] = 0
        df['aprioriY'] = 0
        df['aprioriZ'] = 0
        df['adjustedX'] = 0
        df['adjustedY'] = 0
        df['adjustedZ'] = 0
        df['aprioriCovar'] = [[] for _ in range(len(df))]

        #only populate the new columns for ground points. Otherwise, isis will
        #recalculate the control point lat/lon from control measures which where
        #"massaged" by the phase and template matcher.
        for i, row in df.iterrows():
            if row['pointtype'] == 3 or row['pointtype'] == 4:
                if row['apriori']:
                    apriori_geom = swkb.loads(row['apriori'], hex=True)
                    row['aprioriX'] = apriori_geom.x
                    row['aprioriY'] = apriori_geom.y
                    row['aprioriZ'] = apriori_geom.z
                if row['adjusted']:
                    adjusted_geom = swkb.loads(row['adjusted'], hex=True)
                    row['adjustedX'] = adjusted_geom.x
                    row['adjustedY'] = adjusted_geom.y
                    row['adjustedZ'] = adjusted_geom.z
                df.iloc[i] = row

        return df


def update_from_jigsaw(cnet, measures, connection, pointid_func=None):
    """
    Updates a database fields: liner, sampler, measureJigsawRejected,
    samplesigma, and linesigma using an ISIS control network.
    
    This function uses the pandas update function with overwrite=True. Therefore, 
    this function will overwrite NaN and non-NaN entries.

    In order to be efficient, this func creates an in-memory control network
    and then writes to the database using a string buffer and a COPY FROM call.
    
    Note: If using this func and looking at the updates table in pgadmin, it
    is necessary to refresh the pgadmin table of contents for the schema.

    Parameters
    ----------
    cnet : pd.DataFrame
           plio.io.io_control_network loaded dataframe

    measures : pd.DataFrame
               of measures from a database table. 
    
    connection : object
                 An SQLAlchemy DB connection object

    poitid_func : callable
                  A callable function that is used to split the id string in
                  the cnet in order to extract a pointid. An autocnet written cnet
                  will have a user specified identifier with the numeric pointid as 
                  the final element, e.g., autocnet_1. This func needs to get the
                  numeric ID back. This callable is used to unmunge the id.
    """

    def copy_from_method(table, conn, keys, data_iter, pre_truncate=False, fatal_failure=False):
        """
        Custom method for pandas.DataFrame.to_sql that will use COPY FROM
        From: https://stackoverflow.com/questions/24084710/to-sql-sqlalchemy-copy-from-postgresql-engine
        
        This is follows the API specified by pandas.
        """

        dbapi_conn = conn.connection
        cur = dbapi_conn.cursor()

        s_buf = StringIO()
        writer = csv_writer(s_buf, quoting=QUOTE_MINIMAL)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        table_name = '{}.{}'.format(
            table.schema, table.name) if table.schema else table.name

        sql_query = 'COPY %s (%s) FROM STDIN WITH CSV' % (table_name, columns)
        cur.copy_expert(sql=sql_query, file=s_buf)
        return cur.rowcount

    # Get the PID back from the id.
    if pointid_func:
        cnet['pointid'] = cnet['id'].apply(pointid_func)
    else:
        cnet['pointid'] = cnet['id']
    cnet = cnet.rename(columns={'sampleResidual':'sampler',
                            'lineResidual':'liner'})

    # Homogenize the indices
    measures.set_index(['pointid', 'serialnumber'], inplace=True)
    cnet.set_index(['pointid', 'serialnumber'], inplace=True)

    # Update the current meaasures using the data from the input network
    measures.update(cnet[['sampler', 'liner', 'measureJigsawRejected', 'samplesigma', 'linesigma']])
    measures.reset_index(inplace=True)
    
    # Compute the residual from the components
    measures['residual'] = np.sqrt(measures['liner'] ** 2 + measures['sampler'] ** 2)

    # Execute an SQL COPY from a CSV buffer into the DB
    measures.to_sql('measures_tmp', connection, schema='public', if_exists='replace', index=False, method=copy_from_method)

    # Drop the old measures table and then rename the tmp measures table to be the 'new' measures table
    connection.execute('DROP TABLE measures;')
    connection.execute('ALTER TABLE measures_tmp RENAME TO measures;')

# This is not a permanent placement for this function
# TO DO: create a new module for parsing/cleaning points from a controlnetwork
from scipy.stats import zscore
from plio.io.io_gdal import GeoDataset
from autocnet.io.db.model import Images
import pvl
def null_measure_ignore(point, size_x, size_y, valid_tol, verbose=False, ncg=None, **kwargs):

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    isis2np_types = {
            "UnsignedByte" : "uint8",
            "SignedWord" : "int16",
            "Real" : "float64"}

    resultlog = []
    with ncg.session_scope() as session:
        pid = point.id
        print('point id: ', pid)
        measures = session.query(Measures).filter(Measures.pointid==pid).order_by(Measures.id).all()
        print('number of measures: ', len(measures))
        for measure in measures:
            currentlog = {'measureid': measure.id,
                          'status': 'No change'}
            m_imageid = measure.imageid
            m_image = session.query(Images).filter(Images.id==m_imageid).one()
            cube = GeoDataset(m_image.path)

            center_x = measure.sample
            center_y = measure.line

            start_x = int(center_x - size_x)
            start_y = int(center_y - size_y)
            stop_x = int(center_x + size_x)
            stop_y = int(center_y + size_y)

            pixels = list(map(int, [start_x, start_y, stop_x-start_x, stop_y-start_y]))
            dtype = isis2np_types[pvl.load(cube.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
            arr = cube.read_array(pixels=pixels, dtype=dtype)

            z = zscore(arr, axis=0)
            nn= sum(sum(np.isnan(z)))
            percent_valid = (1 - nn/z.size)*100
            if percent_valid < valid_tol:
                session.query(Measures).\
                        filter(Measures.pointid==pid, Measures.id==measure.id).\
                        update({'ignore': True})
                currentlog['status'] = 'Ignored'

            resultlog.append(currentlog)
    return resultlog

