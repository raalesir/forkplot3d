

# import  pytest
import  numpy as np


from forkplot3d.energies import  FullStretchEnergy, FullIntersizeEnergy, FullHistogramEnergy, FullPoleAttractionEnergy
from forkplot3d.node import  Node

def test_stretch_energy():

    test_c = np.array(range(1,13)).reshape(1,3,4)

    assert FullStretchEnergy.energy(c = test_c, l_0=4)==64



def test_full_intersize_energy():

    test_c = np.array(range(1, 13)).reshape(1, 3, 4)
    assert  FullIntersizeEnergy.energy(c=test_c) == 9



def test_full_histogram_energy():

    test_coords = np.array([[0, 1., 2., 3.], [0, 0, 0, 0], [0, 0, 0, 0]]).reshape(3, 4, 1)
    test_coords += 0.5

    distibutions = []
    for i in range(test_coords.shape[1]):
        vals, bins = np.array([1, 2, 2, 1]), np.array([i, 1 + i, 2 + i, 3 + i])
        distibutions.append(Node(vals, bins, i, 0))
    print(distibutions[0].get_bins(), distibutions[0].get_vals())
    test_array = np.array(distibutions).reshape(4, 1)

    assert FullHistogramEnergy.energy(c=test_coords, array=test_array) == -8


def test_full_pole_attraction_energy1():

    test_coords = np.array([[0, 1., 2., 3.], [0, 0, 0, 0], [0, 0, 0, 0]]).reshape(3, 4, 1)
    test_coords += 0.5

    distibutions = []
    for i in range(test_coords.shape[1]):
        vals, bins = np.array([1, 2, 2, 1]), np.array([i, 1 + i, 2 + i, 3 + i])
        distibutions.append(Node(vals, bins, i, 0))
    print(distibutions[0].get_bins(), distibutions[0].get_vals())
    test_array = np.array(distibutions).reshape(4, 1)

    assert FullPoleAttractionEnergy.energy(c=test_coords, array=test_array, x0=0) == 30


def test_full_pole_attraction_energy2():
    test_coords = np.array([[0, 1., 2., 3.], [0, 0, 0, 0], [0, 0, 0, 0]]).reshape(3, 4, 1)
    test_coords += 0.5

    distibutions = []
    for i in range(test_coords.shape[1]):
        vals, bins = np.array([1,2,2,1]), np.array([0,1,2,3])
        distibutions.append(Node(vals,  bins, i, 0))
    print(distibutions[0].get_bins(), distibutions[0].get_vals())
    test_array = np.array(distibutions).reshape(4,1)

    assert FullPoleAttractionEnergy.energy(c=test_coords, array=test_array, x0=0) == 14
