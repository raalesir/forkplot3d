"""
System under  simulation
"""
import logging
import os
from scipy.io import loadmat
import  pandas as pd
import  sys
import  numpy as np
try:
    from node import Node #https://stackoverflow.com/questions/21236824/unresolved-reference-issue-in-pycharm
    from parameters import *
except ModuleNotFoundError:
    from .node import Node
    from .parameters import *


class System:
    """
    What  is the  system we use to simulate? :)
    The system has coordinates and physical properties like,  energy, volume etc.

    """

    def __init__(self, number_of_size_bins_, number_of_histogram_bins_, *args, **kwargs):
        self.number_of_beads = None  # number of fork-plots
        self.number_of_size_bins = number_of_size_bins_  # number of size bins for a  fork-plot
        self.number_of_histogram_bins = number_of_histogram_bins_  # number of bin in each histogram

        self.files = None  # list of input data files

        self.array = None

        self.coords = None  # the Cartesian coordinates describing system

        self.interactions  = self._register_interactions()



    def __repr__(self):
        return "happy system"


    def _register_interactions(self):
        """
        which types of interactions to use?
        returns  a dictionary with the keys as label for the class and the value is actual class
        :return:
        :rtype:
        """
        import energies  as  e

        d = {}
        energies_ = [elem for elem in dir(e) if 'Energy' in elem]

        for energy in energies_:
            if eval('e.' + energy).USE:
                d[eval('e.' + energy).label] = eval('e.' + energy)

        return d


    def prepare_system(self, datafiles_location, fnames_):

        logging.info('preparing experimental setup')

        logging.info("datafiles are located  at: %s" % (datafiles_location))
        logging.info('the sourse files are:%s' % (fnames_))

        files = [os.path.join(datafiles_location, fname) for fname in fnames_]
        logging.info("checking files for existence...")
        self.files = [file for file in files if os.path.isfile(file)]
        logging.info('done')
        logging.info(files)

        self.number_of_beads = len(self.files)


        self.array = self._get_distributions()

        self.coords = self._get_init_coords()

        return



    def _get_distributions(self):
        """
        Takes in a list of *.mat* files.
        For each  file the area is binned into `self.number_of_size_bins` area bins.
        For  each area bin the distribution  with `self.number_of_histogram_bins` long-axis bins for the long  axis is being produced.

        The long-axis bins along with the corresponding values are constituting an array element.
        The array of dimension `self.number_of_beads, self.number_of_size_bins` is being returned.

        :param files:
        :type files:
        :param n_size_bins:
        :type n_size_bins:
        :param n_bins:
        :type n_bins:
        :return:
        :rtype:
        """

        distributions = []

        # n_forks = len(files)

        for i in range(self.number_of_beads):
            file = self.files[i]
            logging.info('loading %s' %file)
            annots = loadmat(file)
            logging.info('done loading %s' %file)
            logging.info('creating a Pandas DF from the data... ')

            try:
                fork_data = pd.DataFrame.from_dict(data={'areas': list(annots['areas'].T),
                                                         'lengths': list(annots['lengths'].T),
                                                         'longs': list(annots['longs'].T),
                                                         'shorts': list(annots['shorts'].T)
                                                         }
                                                   ).astype('float')

                fork_data['areas_binned'] = pd.cut(fork_data['areas'], bins=self.number_of_size_bins, )
                fork_data['bin_number'] = pd.cut(fork_data['areas'], bins=self.number_of_size_bins, labels=False)
                fork_data['longs'] = fork_data['longs'] * fork_data['lengths']


            except  KeyError:
                logging.error("open the %i and check the column names!" %file)
                sys.exit("open the %i and check the column names!" %file)

            logging.info("DF created, dimensions: %i,%i"%(fork_data.shape[0], fork_data.shape[1]))
            for bin_ in range(self.number_of_size_bins):
                logging.info("creating histogram for bin %i" %bin_)

                tmp = fork_data[fork_data['bin_number'] == bin_]['longs'].dropna()
                logging.debug("data shape is: %s"%tmp.shape)
                logging.debug(tmp.describe())

                vals, bins = np.histogram(tmp, bins=self.number_of_histogram_bins, density=True)
                logging.info("done")
                distributions.append(Node(vals, bins, i, bin_))
        logging.info("creating the output array...")
        array = np.array(distributions).reshape((self.number_of_beads, self.number_of_size_bins))
        logging.info("done. The dimension of array is: (%i,%i)"%(array.shape[0], array.shape[1]))
        logging.debug(self.array)

        return array


    def _get_init_coords(self):
        """
        get  initial coordinates
        """

        c = np.zeros((3, self.number_of_beads, self.number_of_size_bins))

        for j in range(self.number_of_beads):
            for k in range(self.number_of_size_bins):
                # fetching array element
                bins = self.array[j, k].get_bins()[:-1]  # to make sizes of  vals and bins equal
                bin_ = np.random.choice(bins)
                c[0, j, k] = bin_
                c[1, j, k] = np.random.uniform(-l_0, l_0)
                c[2, j, k] = np.random.uniform(-l_0, l_0)

        return c



if __name__ == "__main__":

    system = System(
        number_of_size_bins_=number_of_size_bins,
        number_of_histogram_bins_=number_of_histogram_bins
                    )
    print(system)

    energies = system._register_interactions()
    print(energies)
