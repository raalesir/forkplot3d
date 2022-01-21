

# import  pytest
import  numpy as np


from forkplot3d.energies import  FullStretchEnergy


def test_stretch_energy():

    test_c = np.array(range(1,13)).reshape(1,3,4)

    assert FullStretchEnergy.energy(c = test_c, l_0=4)==64