import numpy as np
from math import pi, sqrt
from random import randint

from pilot.lattice import Lattice2D
from pilot.fields import ClassicalSpinField
from pilot.distributions import spher_to_eucl

L = 4  # must be 4 or greater
TESTING_LATTICE = Lattice2D(L)


def add_minimal_hedgehog(state, loc: tuple, positive: bool = True):
    """Takes a single configuration in the cartesian representation and adds
    a minimal hedgehog Skyrmion centred on "loc".

    Parameters
    ----------
    state: torch.Tensor
        The state to which the hedgehog skyrmion is to be added. Should have
        shape (L, L, 2), with the two components in the last dimension being
        the polar and azimuthal angles, in that order.
    loc: tuple
        The coordinates for the centre of the skyrmion.
    positive: bool
        True means that the added skyrmion will have a topological charge of
        +1 if placed in a background configuration with all polar angles
        being pi and azimuthal angles being 0. False is the reverse case: a
        charge of -1 in a background configuration with all angles being 0.

    Returns
    -------
    state: torch.Tensor
        The input state with the 9 elements centered on "loc" changed to
        a minimal hedgehog skyrmion configuration.

    Notes
    -----
    Totally overkill for these tests. More useful if we wanted to run tests
    with multiple skyrmions, but I can't yet see a need.
    """
    x0, y0 = loc
    if positive:
        sign = 1
    else:
        sign = -1

    polar = np.zeros((3, 3))
    azimuth = np.zeros((3, 3))

    polar[1, 1] = pi * (1 - sign) / 2
    polar[0, 1] = polar[1, 0] = polar[2, 1] = polar[1, 2] = pi / 2
    polar[0, 0] = polar[0, 2] = polar[2, 0] = polar[2, 2] = pi * (
        (1 - sign) / 2 + sign / sqrt(2)
    )

    azimuth[0, 2] = pi / 4
    azimuth[0, 1] = pi / 2
    azimuth[0, 0] = 3 * pi / 4
    azimuth[1, 0] = pi
    azimuth[2, 0] = 5 * pi / 4
    azimuth[2, 1] = 3 * pi / 2
    azimuth[2, 2] = 7 * pi / 4

    # Ensures this works even when loc is on a boundary
    state = np.roll(state, (1 - x0, 1 - y0), (0, 1))
    state[:3, :3, 0] = polar
    state[:3, :3, 1] = azimuth
    state = np.roll(state, (x0 - 1, y0 - 1), (0, 1))

    return state


def test_uncharged():
    theta = np.random.rand(L ** 2) * 0.5 * pi  # all spins sit in one hemisphere
    phi = np.random.rand(L ** 2) * 2 * pi
    state = spher_to_eucl(np.stack((theta, phi), axis=-1))
    field = ClassicalSpinField(state, TESTING_LATTICE)
    assert abs(float(field.topological_charge)) < 1e-7


def test_minimal_hedgehog_pos():
    state = np.stack((np.ones((L, L)) * pi, np.zeros((L, L))), axis=-1)
    state = add_minimal_hedgehog(
        state, (randint(0, L - 1), randint(0, L - 1)), positive=True
    )

    state = spher_to_eucl(state.reshape(-1, 2))

    field = ClassicalSpinField(state, TESTING_LATTICE)
    assert abs(float(field.topological_charge) - 1) < 1e-7


def test_minimal_hedgehog_neg():
    state = np.zeros((L, L, 2))
    state = add_minimal_hedgehog(
        state, (randint(0, L - 1), randint(0, L - 1)), positive=False
    )

    state = spher_to_eucl(state.reshape(-1, 2))

    field = ClassicalSpinField(state, TESTING_LATTICE)
    assert abs(float(field.topological_charge) + 1) < 1e-7
