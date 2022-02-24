"""Main module."""

import logging
from pathlib import Path
import  os

import  matplotlib.pyplot as plt

try:
    from system import  System
    from parameters import *
    from energies import  *

except ModuleNotFoundError:
    from .system import System
    from .parameters import *
    from .energies import *



class Simulation:
    """
    performs a simulation of a system of connected beads.


    """

    def __init__(self, system_, *args, **kwargs):

        self.system = system_


    def run(self, n_steps=10000):
        size_to_monitor = 3

        logging.info("running simulation")

        # e1 = kappa*get_stretch_energy(x)
        # e2 = hist_energy(x, histograms)
        # energy = e1 + e2
        # print(e1, e2, energy)

        c = self.system._get_init_coords()
        # c = init_coords(array=array, l_0=l_0)
        x0 = 4

        kappa = .2
        epsilon = 0.1
        alpha = 0.02
        e1 = FullHistogramEnergy.energy(c=c, array=self.system.array) #full_histogram_energy(c, array)
        e2 = kappa * FullStretchEnergy.energy(c=c, l_0=l_0) #full_stretch_energy(c, l_0=l_0)
        e3 = epsilon * FullIntersizeEnergy.energy(c=c) #intersize_energy(c)
        e4 = alpha * FullPoleAttractionEnergy.energy(c=c, array=self.system.array, x0=x0) #pole_attraction_energy(c, array, x0)
        energy = e1 + e2 + e3 + e4
        logging.info('initial energy: %2.3f, %2.3f,%2.3f,%2.3f,total: %2.3f '%(e1, e2, e3, e4, energy))

        e = [energy]
        bond_length = []
        beta = 10
        take_bonds = 0
        collect_coords = []
        logging.info('running %i cycles' %n_steps)

        for i in range(n_steps):
            fork = np.random.choice(self.system.array.shape[0])
            layer = np.random.choice(self.system.array.shape[1])
            #     layer= 2
            bead = self.system.array[fork, layer]
            #     new_index = np.random.choice(n_bins)
            #                 c[0, fork, layer] = bin_



            c_new = c.copy()
            cur_index = np.where(bead.get_bins() == c_new[0, fork, layer])[0][0]
            new_index = np.random.choice([cur_index - 1, cur_index + 1])

            if new_index >= len(bead.get_bins()) - 1:
                new_index -= 2
            elif new_index <= 0:
                new_index += 2

            c_new[0, fork, layer] = bead.get_bins()[new_index]
            c_new[1, fork, layer] += np.random.uniform(-l_0 / 5, l_0 / 5)
            c_new[2, fork, layer] += np.random.uniform(-l_0 / 5, l_0 / 5)

            if (c_new[1:3, fork, layer] > 1.0).any() | (c_new[1:3, fork, layer] < -1.0).any():
                new_energy = 10000
            else:

                e1_new = FullHistogramEnergy.energy(c=c_new, array=self.system.array)# full_histogram_energy(c_new, array)
                e2_new = kappa * FullStretchEnergy.energy(c=c_new, l_0=l_0) #full_stretch_energy(c_new, l_0=l_0)
                e3_new = epsilon * FullIntersizeEnergy.energy(c=c_new)# intersize_energy(c_new)
                e4_new = alpha * FullPoleAttractionEnergy.energy(c=c_new, array=self.system.array, x0=x0)  #pole_attraction_energy(c_new, array, x0)
                #     energy = e1+e2+e3
                #     i_h = np.random.choice(len(histograms))


                #     h = histograms[i_h]
                #     ind = np.random.choice(n_bins)

                #     x_new = x.copy()

                #     x_new[0,i_h] = h[1][ind]
                #     x_new[1,i_h] = x_new[1,i_h] + np.random.uniform(-l_0/5, l_0/5)
                #     x_new[2,i_h] = x_new[2,i_h] + np.random.uniform(-l_0/5, l_0/5)


                #     e1 = kappa*get_stretch_energy(x_new, histograms)
                #     e2 = hist_energy(x_new, histograms)

                #     i_h_neigbours = get_neighbours(i_h, len(histograms)) #np.arange(len(histograms))

                #     d_e2 = diff_hist_energy(x_new, histograms, i_h_neigbours) - diff_hist_energy(x, histograms, i_h_neigbours)
                #     e2_new = e2 + d_e2

                #     d_e1 = diff_stretch_energy(x_new, i_h_neigbours) - diff_stretch_energy(x, i_h_neigbours)
                #     e1_new = e1 + kappa*d_e1


                new_energy = e1_new + e2_new + e3_new + e4_new

            if np.random.random() < np.exp(-beta * (new_energy - energy)):
                e2 = e2_new
                e1 = e1_new
                e3 = e3_new
                e4 = e4_new
                energy = new_energy

                c = c_new.copy()
                #         x = x_new.copy()
                #         print(sum(energy), sum(e_new), np.exp(-50*(sum(e_new) - sum(energy))))
                e.append(energy)

                x_d = np.diff(c[:, :, size_to_monitor], axis=1)
                bonds = [np.sqrt(np.dot(el, el)) for el in x_d.T]
                bonds = sum(bonds) / len(bonds)
                bond_length.append(bonds)

                if np.random.random() < .05:
                    print(e1, e2, e3, e4)

                    # scatter.x = c[0, :, size_to_monitor]
                    # scatter.y = c[1, :, size_to_monitor]
                    # scatter.z = c[2, :, size_to_monitor]
                    # lines.x = np.append(c[0, :, size_to_monitor], c[0, :1, size_to_monitor])
                    # lines.y = np.append(c[1, :, size_to_monitor], c[1, :1, size_to_monitor])
                    # lines.z = np.append(c[2, :, size_to_monitor], c[2, :1, size_to_monitor])
                    #
                    # scatter1.x = c[0, 2:3, size_to_monitor]
                    # scatter1.y = c[1, 2:3, size_to_monitor]
                    # scatter1.z = c[2, 2:3, size_to_monitor]
                if (i > n_steps // 3) and (np.random.random() < 0.1):
                    collect_coords.append(c)

        # lines.x = x[0]
        #         lines.y = x[1]
        #         lines.z = x[2]
        #         sleep(.01)
        return np.array(collect_coords)


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

    # simulation.plot_forkplot(fnames)

    collected_coords = simulation.run()
    logging.info('collected coords shape is: %s'%str(collected_coords.shape))


