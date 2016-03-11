from time import gmtime, strftime

import pandas as pd

POINT_TYPE = 2
MEASURE_TYPE = 2


class CSeries(pd.Series):
    """
    A custom pandas series that can accept additional methods
    """
    @property
    def _constructor(self):
        return CSeries  # pragma: no cover


class C(pd.DataFrame):
    """
    Control network designed in the ISIS format.

    Attributes
    ----------

    n : int
        Number of control points

    m : int
        Number of control measures

    creationdate : str
                   The date that this control network was created.

    modifieddate : str
                   The date that this control network was last modified.

    Examples
    --------
    This example illustrates the manual creation of a pandas dataframe with
    a multi-index (created from a list of tuples).

    >>> ids = ['pt1','pt1', 'pt1', 'pt2', 'pt2']
    >>> ptype = [2,2,2,2,2]
    >>> serials = ['a', 'b', 'c', 'b', 'c']
    >>> mtype = [2,2,2,2,2]
    >>> multi_index = pd.MultiIndex.from_tuples(list(zip(ids, ptype, serials, mtype)),\
                                    names=['Id', 'Type', 'Serial Number', 'Measure Type'])
    >>> columns = ['Random Number']
    >>> data_length = 5
    >>> data = np.random.randn(data_length)
    >>> C = control.C(data, index=multi_index, columns=columns)

    """

    def __init__(self, *args, **kwargs):
        super(C, self).__init__(*args, **kwargs)
        self._creationdate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @property
    def _constructor(self):
        return C

    _constructor_sliced = CSeries

    @property
    def n(self):
        if not getattr(self, '_n', None):
            self._n = len(self['pid'].unique())
        return self._n

    @property
    def m(self):
        if not getattr(self, '_m', None):
            self._m = len(self)
        return self._m

    @property
    def creationdate(self):
        return self._creationdate

    @property
    def modifieddate(self):
        if not getattr(self, '_modifieddate', None):
            self._modifieddate = 'Not modified'
        return self._modifieddate

    '''
    @modifieddate.setter
    def update_modifieddate(self):
        self._modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    '''
