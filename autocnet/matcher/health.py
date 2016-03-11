

class EdgeHealth(object):
    """
    Storage and computation of the health of a graph edge using the metric:


    """

    def __init__(self):
        self.FundamentalMatrix = 0.0

    @property
    def health(self):
        return self.recompute_health()

    def update(self, *args, **kwargs):
        """
        Pass through called when the observable (model) changes.
        *args and **kwargs are passed through from the observable.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def recompute_health(self):
        """
        Recompute the health of the edge.
        """
        return self.FundamentalMatrix.error.mean()
