"""
distributions.py

Module containing distributions used for generating fields.
"""
import numpy as np
from numpy.random import random

from pilot.utils import spher_to_eucl


class SphericalUniformDist:
    # TODO this should probably return euclidean vectors!!!
    def __init__(self, dim: int):
        self.dim = dim
        self.rng = np.random.default_rng().uniform

        if self.dim < 2:
            raise NotImplementedError
        if self.dim == 2:
            self.rvs = self.gen_circular
        elif self.dim == 3:
            self.rvs = self.gen_spherical
        else:
            raise NotImplementedError

    def gen_circular(self, size: (int, tuple) = 1, insert_at_dim: int = 1):
        """Generates random vectors in R^2 distributed uniformly on the unit circle.
        
        Parameters
        ----------
        size: (int, tuple)
            The size of the output array dimensions, exclusing the vector dimension.
        insert_at_dim: int
            Position at which to insert the vector dimension.

        Returns
        -------
        out: numpy.ndarray
            The array of vectors, with dimension (*, 2, *) where the dimensions either
            side of the vector dimension are specifed by `size` and `insert_at_dim`.
        """
        polar = self.rng(low=0, high=2 * np.pi, size=size)
        return spher_to_eucl(np.expand_dims(polar, axis=1))

    def gen_spherical(self, size: (int, tuple) = 1):
        """Generates random vectors in R^2 distributed uniformly on the unit 2-sphere.
        
        Parameters
        ----------
        size: (int, tuple)
            The size of the output array dimensions, exclusing the vector dimension.
        insert_at_dim: int
            Position at which to insert the vector dimension.

        Returns
        -------
        out: numpy.ndarray
            The array of vectors, with dimension (*, 2, *) where the dimensions either
            side of the vector dimension are specifed by `size` and `insert_at_dim`.
        """
        polar = np.arccos(1 - 2 * self.rng(size=size))
        azimuth = self.rng(low=0, high=2 * np.pi, size=size)

        return spher_to_eucl(np.stack((polar, azimuth), axis=1))
