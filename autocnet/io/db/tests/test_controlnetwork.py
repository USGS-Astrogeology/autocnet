import sys

import pytest
from autocnet.io.db import model
from autocnet.io.db.controlnetwork import db_to_df

if sys.platform.startswith("darwin"):
    pytest.skip("skipping DB tests for MacOS", allow_module_level=True)

def test_to_isis(session, db_controlnetwork):
    df = db_to_df(session.get_bind())

    assert len(df) == 6
    assert df.iloc[0]['pointtype'] == 2
    assert df.iloc[4]['pointtype'] == 3
    assert df.iloc[0]['measuretype'] == 3
    assert df.iloc[0]['aprioriCovar'] == []
