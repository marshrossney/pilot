"""
generators.py
"""
import numpy as np
from numpy.random import random


def spher_to_eucl(coords):
    """Converts a set (N-1) angles to a set of N-component euclidean unit vectors.

    # TODO
    The order of the (N-1) angles [\phi^0, ..., \phi^{N-1}] is taken to match some
    convention.

    Parameters
    ----------
    coords: numpy.ndarray
        The spherical coordinates (angles). The (N-1) angles are expected on the 1st
        dimension. Dimension (lattice.volume, (N-1), *).

    Returns
    -------
    out: numpy.ndarray
        The Euclidean representation of the angles, dimension (lattice.volume, N, *).

    Notes
    -----
    See REF
    """
    output_shape = list(coords.shape)
    output_shape[1] += 1

    output = np.ones(output_shape)
    output[:, :-1] = np.cos(coords)
    output[:, 1:] *= np.cumprod(np.sin(coords), axis=1)
    return output


class SphericalUniformDist:
    """
    Methods
    -------
    rvs: generates random variates according to the spherical uniform distribution
         with dimension `dim`. Nomenclature chosen for consistency with scipy.distributions.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.rng = np.random.default_rng().uniform

        if self.dim < 2:
            raise ValueError("Dimension cannot be less than 2")
        if self.dim == 2:
            self.rvs = self.gen_circular
        elif self.dim == 3:
            self.rvs = self.gen_spherical
        else:
            raise NotImplementedError

    def gen_circular(
        self, size: (int, tuple) = 1, delta_max: int = 1, insert_at_dim: int = 1
    ):
        """Generates random vectors in R^2 distributed uniformly on the unit circle.
        
        Parameters
        ----------
        size: (int, tuple)
            The size of the output array dimensions, exclusing the vector dimension.
        delta_max: int
            
        vector_dim: int
            Position at which to insert the vector dimension.

        Returns
        -------
        out: numpy.ndarray
            The array of vectors, with dimension (*, 2, *) where the dimensions either
            side of the vector dimension are specifed by `size` and `vector_dim`.
        """
        polar = self.rng(low=0, high=2 * np.pi, size=size)
        return spher_to_eucl(np.expand_dims(polar, axis=insert_at_dim))

    def gen_spherical(self, size: (int, tuple) = 1, vector_dim: int = 1):
        """Generates random vectors in R^2 distributed uniformly on the unit 2-sphere.
        
        Parameters
        ----------
        size: (int, tuple)
            The size of the output array dimensions, exclusing the vector dimension.
        vector_dim: int
            Position at which to insert the vector dimension.

        Returns
        -------
        out: numpy.ndarray
            The array of vectors, with dimension (*, 2, *) where the dimensions either
            side of the vector dimension are specifed by `size` and `vector_dim`.
        """
        polar = np.arccos(1 - 2 * self.rng(size=size))
        azimuth = self.rng(low=0, high=2 * np.pi, size=size)

        return spher_to_eucl(np.stack((polar, azimuth), axis=vector_dim))
