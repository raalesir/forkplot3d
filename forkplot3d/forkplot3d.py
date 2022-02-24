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


    def plot_forkplot(self, fnames):
        """
        plotting fork-plots for all sizes
        and saves the result to the `plots` folder
        """


        for s in range(self.system.array.shape[1]):
            logging.info('producing fork plot for size %i out of %i'%(s, self.system.array.shape[1]))
            plt.figure(figsize=(18, 12))
            COLOR = 'black'
            plt.rcParams['text.color'] = COLOR
            plt.rcParams['axes.labelcolor'] = COLOR
            plt.rcParams['xtick.color'] = COLOR
            plt.rcParams['ytick.color'] = COLOR
            plt.suptitle("Fork-plots for size bin %i out of %i" % (s + 1, self.system.array.shape[1]), y=0.94, fontsize=25)
            for i in range(self.system.array.shape[0]):
                plt.subplot(self.system.array.shape[0] // 2, 2, i + 1)
                v = self.system.array[i, s]
                plt.bar(v.get_bins()[:-1], v.get_vals(), width=0.1,
                        label='fork=%i, file=%s' % (i + 1, fnames[i]))
                plt.xlim(0, 4)
                plt.xticks(fontsize=20)
                plt.xlabel("position, long axis, $\mu m$", fontsize=20)
                plt.legend(fontsize=16, framealpha=0.1)

            path = os.path.join(PLOT_ROOT, 'forkplot_size_%i.png' % s)
            plt.savefig(path, transparent=True, bbox_inches='tight')



    def sample_random_configurations(self, fname):
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
                e += func.energy(c=c, array=self.system.array, l_0=l_0, x0=0 )

            energies.append(e)

        plt.title("Energy distribution for %i random conformations" %n_sample)
        plt.xlabel('energy, units')
        plt.ylabel('population')
        plt.hist(energies, edgecolor='black')

        Path(PLOT_ROOT).mkdir(parents=True, exist_ok=True)
        path = os.path.join(PLOT_ROOT, fname)
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

    logging.info("the distribution shape for the first array element: %i " %(system.array[0,0].get_vals().shape[0], ))

    simulation = Simulation(system)


    # logging.info("sampling  some random conformations and plotting full energy distributions...")
    # logging.info("the plot is saved to  %s" %(os.path.join(PLOT_ROOT,'random.png')))
    # simulation.sample_random_configurations('random.png')

    simulation.plot_forkplot(fnames)

    # simulation.run()

