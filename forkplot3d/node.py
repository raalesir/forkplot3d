class Node:
    """
    class representing  a distribution
    """

    def __init__(self, vals, bins, i, j, *args, **kwargs):
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