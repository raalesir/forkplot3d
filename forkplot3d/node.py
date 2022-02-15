class Node:
    """
    class representing  a distribution, given by `vals` as values and `bins` as bins for the values.
    Each instance has a position: `i` is the  index of fork plot, and `j` is the index of size bin.
    """

    def __init__(self, vals, bins, i, j, *args, **kwargs):
        """

        :param vals: values for the fork plot
        :type vals: numpy array
        :param bins: bins for the fork plot
        :type bins:  numpy array
        :param i: index for the fork plot
        :type i:int
        :param j: index for the size bin
        :type j: int

        """
        self.vals = vals
        self.bins = bins
        self.i = i
        self.j = j

    def __repr__(self):
        return "p(%s,%s)" % (self.i, self.j)

    def get_vals(self):
        return self.vals

    def get_bins(self):
        return self.bins



if __name__ == "__main__":

    node = Node([1,2,3], [1,2,3], 0,0)
    print(node)