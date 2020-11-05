from pathlib import PosixPath
from psycopg2.extensions import register_adapter, AsIs, adapt

# Adaptors as defined by https://www.psycopg.org/docs/advanced.html#adapting-new-python-types-to-sql-syntax

def adapt_posixpath(posixpath):
    stringified = adapt(str(posixpath))
    return stringified

# TODO: Add adaptors from numpy types. We shouldn't have to typecast 
#  explicitly across the code base.

register_adapter(PosixPath, adapt_posixpath)