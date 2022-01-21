"""Main module."""

import logging
from pathlib import Path
import  os

import  matplotlib.pyplot as plt

try:
    from system import  System
    from parameters import *

except ModuleNotFoundError:
    from .system import System
    from .parameters import *


class Simulation:
    """
    performs a simulation of a system of connected beads.


    """

    def __init__(self, system_, *args, **kwargs):

        self.system = system_


    def run(self):
        logging.info("running simulation")


        return


    def sample_random_configurations(self):
        """
        generate random comformation, calculate the energy, accumulate it and
        plot energy distribution.
        The plot is saved to the disk.

        :return: plot saved to
        :rtype:
        """

        energies = []
        n_sample = 1000
        for i in range(n_sample):
            c = self.system._get_init_coords()

            e = 0
            for label, func in self.system.interactions.items():
                e += func.energy(c=c, array=self.system.array, l_0=l_0 )

            energies.append(e)

        plt.title("Energy distribution for %i random conformations" %n_sample)
        plt.xlabel('energy, units')
        plt.ylabel('population')
        plt.hist(energies, edgecolor='black')

        Path(PLOT_ROOT).mkdir(parents=True, exist_ok=True)
        path = os.path.join(PLOT_ROOT, 'random.png')
        plt.savefig(path)



if __name__ == "__main__":


    prefix = '/Users/alexey/Downloads/'

    # the order  is important!
    fnames = ['0.5MbpFromOri_leftArm.mat', '1MbpFromOri_rightArm.mat',
              '2.1MbpFromOri_rightArm.mat', '1.1MbpFromOri_leftArm.mat'
              ]


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    logging.info('=='*50)


    system  = System(
        number_of_size_bins_=number_of_size_bins,
        number_of_histogram_bins_=number_of_histogram_bins
    )

    logging.info("registered interactions are: %s" %str(system.interactions))

    system.prepare_system(datafiles_location=prefix, fnames_=fnames)


    logging.info("systems' coordinates shape is: %s"%(str(system.coords.shape )))

    simulation = Simulation(system)


    simulation.sample_random_configurations()
    # simulation.run()

