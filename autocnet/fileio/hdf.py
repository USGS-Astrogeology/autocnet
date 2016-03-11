import h5py as h5


class HDFDataSet(object):
    """
    Read / Write an HDF5 dataset using h5py
    """

    # TODO: This is dumb, why did I hard code this...
    def __init__(self, filename='/scratch/jlaura/newrun.h5'):
        self.filename = filename
        self.groups = None

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = h5.File(self.filename)
        return self._data

    def getgroups(self):
        """
        Get all of the first order neighbors to the root node.

        Returns
        -------
        groups : list
            A unicode list of the keys of the file.
        """
        if self.groups is None:
            self.groups = self.data.keys()
        return self.groups

        def getattributes(self):
            if self.groups is None:
                self.groups = self.data.keys()

            for k in self.groups:
                print(self.data[k].attrs.items())
