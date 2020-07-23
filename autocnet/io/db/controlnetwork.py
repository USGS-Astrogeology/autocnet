import pandas as pd
import shapely.wkb as swkb


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
