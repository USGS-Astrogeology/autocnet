import numpy as np
import pandas as pd

from pandas.core.indexing import _LocIndexer


class DbDataFrame(pd.DataFrame):
    """
    Lightweight subclass of a pandas dataframe that overrides
    __setitem__ to get changes pushed back into a database. The
    class functions by calling the parent's (pd.DataFrame) 
    __setitem__ (thereby using all of the Pandas logic) and then
    calling whatever setter might be defined.
    """
    _metadata = ['parent', 'name']

    def __init__(self, *args, **kwargs):
        parent = kwargs.pop('parent', None)
        name = kwargs.pop('name', None)
        super(DbDataFrame, self).__init__(*args, **kwargs)
        self.parent = parent
        self.name = name

    @property
    def _constructor(self):
        return DbDataFrame

    @property	  	
    def loc(self, *args, **kwargs):      	
        """See pandas.Index.loc; preserves metadata"""
        if self._loc is None:
            try:
                self._loc = _LocIndexer(self)
            #New versions of _IXIndexer require "name" attribute.
            except TypeError as TE:
                self._loc= _LocIndexer(self, 'loc')
        return self._loc 

    def __setattr__(self, name, value):
        
        super(DbDataFrame, self).__setattr__(name, value)
        #print(name, type(self), self)
        if name == '_data':            
            setattr(self.parent, self.name, self)

    def __setitem__(self, key, value):
        super(DbDataFrame, self).__setitem__(key, value)
        setattr(self.parent, self.name, self)

    @property
    def _constructor_sliced(self):
        return DbSeries

class DbSeries(pd.Series):
    _metadata = ['parent', 'name']

    def __new__(cls, *args, **kwargs):
        kwargs.pop('parent', None)
        kwargs.pop('name', None)

        arr = pd.Series.__new__(cls)
        if type(arr) is DbSeries:
            return arr
        else:
            return arr.view(DbSeries)

    def __init__(self, *args, **kwargs):
        super(DbSeries, self).__init__(*args, **kwargs)



    @property
    def _constructor(self):
        return DbSeries

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(super(DbSeries, self), mtd)(*args, **kwargs)
        if type(val) == pd.Series:
            val.__class__ = DbSeries
            val.parent = self.parent
            val.name = self.name
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method('__getitem__', key)

    def __setitem__(self, key, value):
        return self._wrapped_pandas_method('__setitem__', key, value)
        #object.__setitem__(key, value)
    
    def __setattr__(self, key, value):
        res =  self._wrapped_pandas_method('__setattr__', key, value)
        return res
