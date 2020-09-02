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


def update_measure_from_jigsaw(point, path, ncg=None, **kwargs):
    """
    Updates the database (associated with ncg) with a single measure's
    jigsaw line and sample residuals.

    Parameters
    ----------
    point   : obj
              point identifying object as defined by autocnet.io.db.model.Points

    path    : str
              absolute path and network name of control network used to update the measure/database.

    ncg     : obj
              the network candidate graph associated with the measure/database
              being updated.
    """

    if not ncg.Session:
        BrokenPipeError('This function requires a database session from a NetworkCandidateGraph.')

    data = cnet.from_isis(path)
    data_to_update = data[['id', 'serialnumber', 'measureJigsawRejected', 'sampleResidual', 'lineResidual', 'samplesigma', 'linesigma', 'adjustedCovar', 'apriorisample', 'aprioriline']]
    data_to_update.loc[:,'adjustedCovar'] = data_to_update['adjustedCovar'].apply(lambda row : list(row))
    data_to_update.loc[:,'id'] = data_to_update['id'].apply(lambda row : int(row))

    res = data_to_update[(data_to_update['id']==point.id)]
    if res.empty:
        print(f'Point {point.id} does not exist in input network.')
        return

    # update
    resultlog = []
    with ncg.session_scope() as session:
        for row in res.iterrows():
            row = row[1]
            currentlog = {'measure':row["serialnumber"],
                          'status':''}

            residual = np.linalg.norm([row["sampleResidual"], row["lineResidual"]])
            session.query(Measures).\
                    filter(Measures.pointid==point.id, Measures.serial==row["serialnumber"]).\
                    update({"jigreject": row["measureJigsawRejected"],
                        "sampler": row["sampleResidual"],
                        "liner": row["lineResidual"],
                        "residual": residual,
                        "samplesigma": row["samplesigma"],
                        "linesigma": row["linesigma"],
                        "apriorisample": row["apriorisample"],
                        "aprioriline": row["aprioriline"]})
            currentlog['status'] = 'success'
            resultlog.append(currentlog)

        session.commit()
    return resultlog
