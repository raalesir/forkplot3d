"""
module for  keeping implementation of the  interactions for the system
"""

import numpy as np
try:
    from  parameters import *
except ModuleNotFoundError:
    from .parameters import * ### needed for pytest.... :((((((

class FullIntersizeEnergy:
    """
    energy associated  with the distance between X-coordinate in different  size bins

    """

    USE = True
    label = 'full_intersize_energy'

    @staticmethod
    def energy(*args, **kwargs):

        c = kwargs['c']

        diffs = np.diff(c[0, :, :], n=1, axis=1)
        return np.sum(np.abs(diffs))


class AnotherEnergy(FullIntersizeEnergy):

    USE = False
    label = 'full_another_energy'




class FullStretchEnergy:

    USE = True
    label = 'full_stretch_energy'

    @staticmethod
    def energy(*args, **kwargs):

        c = kwargs['c']
        l_0 =  kwargs['l_0']

        tmp = np.diff(np.hstack((c, c[:, :1, :])), n=1, axis=1)

        el = 0
        n_forks = c.shape[1]
        n_size_bins = c.shape[2]
        for j in range(n_forks):
            for k in range(n_size_bins):
                t = np.sqrt(np.dot(tmp[:, j, k], tmp[:, j, k])) - l_0
                el += t ** 2
        return el  # , kappa*np.linalg.norm(tmp)




class FullHistogramEnergy:

    USE=True
    label = 'full_histogram_energy'

    @staticmethod
    def energy(*args, **kwargs):

        """
        returns a histogram energy
        """
        c = kwargs['c']
        array = kwargs['array']

        e = 0
        n_forks = c.shape[1]
        n_size_bins = c.shape[2]
        for j in range(n_forks):
            for k in range(n_size_bins):
                vals, bins = array[j, k].get_vals(), array[j, k].get_bins()
                ind = np.argwhere(bins >= c[0, j, k])[0][0]
                e += vals[ind]

        return -e



class FullPoleAttractionEnergy:
    USE= True
    label = 'full_pole_energy'

    @staticmethod
    def energy(*args, **kwargs):
        """
        returns pole attraction energy

        """
        c = kwargs['c']
        array = kwargs['array']
        x0 = kwargs.pop('x0')


        e = 0
        n_forks = c.shape[1]
        n_size_bins = c.shape[2]
        for j in range(n_forks):
            for k in range(n_size_bins):
                vals, bins = array[j, k].get_vals(), array[j, k].get_bins()
                ind = np.argwhere(bins >= c[0, j, k])
                if ind.size > 0:  # checking for empty array
                    ind = ind[0][0]
                    #                 print(ind, bins[ind])
                    e += (bins[ind] - x0) ** 2
        return e