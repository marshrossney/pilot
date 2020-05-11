import numpy as np
from random import random, randint
from math import pi, exp, sqrt
import math as m

import scipy.stats
from pilot.distributions import SphericalUniformDist
from pilot.utils import (
    NotDefinedForField,
    bootstrapped,
    unit_norm,
    requires,
    string_summary,
)


class HamiltonianMismatchError(Exception):
    pass


class NotEnsembleError(Exception):
    pass


class requires_topology(requires):
    attributes = ("has_topology",)
    exception = NotDefinedForField
    message = "Topology not defined for field"


class requires_ensemble(requires):
    attributes = ("is_ensemble",)
    exception = NotEnsembleError
    message = "Property is only defined for ensembles"


class Field:
    """
    Base class for field objects
    """

    minimum_dimensions = 1

    def __init__(self, input_coords, lattice, **theory_kwargs):

        self.lattice = lattice  # must do this before setting coords!
        self.coords = input_coords

        # Unpack theory kwargs
        self.__dict__.update(theory_kwargs)

        self.shift = self.lattice.get_shift()

        self.has_spins = False
        self.has_topology = False

    def __str__(self):
        return string_summary(self, "Field")

    def _valid_field(self, array_in):
        # Check that 0th dimension of input data matches the number of lattice sites
        assert (
            array_in.shape[0] == self.lattice.volume
        ), f"Size of coordinates array at dimension 0: {array_in.shape[0]} does not match volume of lattice: {instance.lattice.volume}"
        # Check that the input data has at least the minimum number of dimensions
        n_dims = len(array_in.shape)
        assert (
            n_dims >= self.minimum_dimensions
        ), f"Invalid number of dimensions in coordinate array. Expected {instance.minimum_dimensions} or more, but found {n_dims}."
        # If missing, add ensemble dimension for convenience
        if n_dims == self.minimum_dimensions:
            array_in = np.expand_dims(array_in, axis=-1)

        return array_in

    @property
    def coords(self):
        """The set of coordinates which define the configuration or ensemble.
        numpy.ndarray, dimensions (lattice_volume, N, *, ensemble_size)"""
        return self._coords

    @coords.setter
    def coords(self, new):
        """Setter for coords which performs some basic checks (see _valid_field)."""
        self._coords = self._valid_field(new)

    @property
    def is_ensemble(self):
        """Bool indicating whether the coordinates have an ensemble dimension with a
        size greater than one."""
        if len(self.coords.squeeze().shape) >= self.minimum_dimensions + 1:
            return True
        return False


class ClassicalSpinField(Field):
    """
    Field class for classical spin fields.

    Parameters
    ----------
    input_coords: numpy.ndarray
        The coordinates of the spin configuration or ensemble.
        Valid inputs may have dimensions:
            - (lattice_volume, N)
            - (lattice_volume, N, ensemble_size)
            - (lattice_volume, N, *, ensemble_size)
        where N is the Euclidean dimension of the spins.
        The final dimension will always be treated as the ensemble dimension in the
        context of calculating ensemble averages.
    lattice: pilot.lattice.Lattice
        Lattice object upon which the spins are defined. Must have the same number of
        sites as the 0th dimension of `input_coords`.
    beta: float
        A theory parameter specifying the coupling strength for interactions between
        spins at different lattice sites. The inverse of the temperature.

    Class attributes
    ----------------
    generators: dict
        TODO

    Notes
    -----
    (1) The lattice field theory is assumed to possess a translational symmetry in the
        directions specified by the primitive lattice vectors. This allows the definition
        of a two point correlation function that is a function of separations only (not
        absolute coordinates) by averaging over the lattice sites.
        return self._calc_vol_avg_two_point_correlator(self.spins)
    """

    minimum_dimensions = 2
    summary_properties = [
        "euclidean_dimension",
        "beta",
    ]

    def __init__(self, input_coords, lattice, beta=1.0):
        super().__init__(input_coords, lattice, beta=beta)

        self.spins = self.coords
        self.euclidean_dimension = self.spins.shape[1]

        self.has_spins = True
        if self.euclidean_dimension == 3:
            self.has_topology = True

        # Global shift in O(N) action (only important for absolute quantities)
        self._action_shift = self.beta * len(self.lattice.dimensions) * self.lattice.volume

    @classmethod
    def new_like(cls, input_coords, template, input_spherical=False):
        """Returns a new instance of the class with the same lattice and theory parameters,
        but with a new set of coordinates.
        
        Parameters
        ----------
        input_coords: numpy.ndarray
            The new coordinates for the field or ensemble of fields. Dimensions
            (lattice.volume, N, *).
        template: Field
            The Field object whose lattice and theory parameters will be used to instantiate
            the new object.
        input_spherical: bool
            If true, input_coords are assumed to be a set of (N-1) angles, and are converted
            to Euclidean vectors. See utils.spher_to_eucl.
        """
        return cls(input_coords, template.lattice, beta=template.beta)

    @classmethod
    def from_random(cls, lattice, *, N, ensemble_size=1, **theory_kwargs):
        input_coords = SphericalUniformDist(dim=N).rvs(
            size=(lattice.volume, ensemble_size)
        )
        return cls(input_coords, lattice, **theory_kwargs)

    def _hamiltonian(self, spins):
        """Calculates the Hamiltonian for a classical spin system or ensemble."""
        return -np.sum(
            spins[self.shift] * np.expand_dims(spins, axis=1),
            axis=2,  # sum over vector components
        ).sum(
            axis=0,  # sum over dimensions
        ).sum(
            axis=0,  # sum over volume
        )

    def _magnetisation_sq(self, spins):
        """Calculates the square of the magnetisation for a spin system or ensemble."""
        return np.sum(
            spins.sum(axis=0) ** 2, axis=0
        )  # sum over volume, then vector components

    def _vol_avg_two_point_correlator(self, spins):
        """Calculates the volume-averaged two point connected correlation function for an
        ensemble of field configurations."""
        _, _, *extra_dims = spins.shape
        # Take positive diagonal shifts only to save time
        n_pos_diag = min(self.lattice.dimensions) // 2 + 1
        va_correlator = np.empty((n_pos_diag, *extra_dims))

        # Disconnected part
        for vector, shift in self.lattice.two_point_iterator(
            pos_only=True, diag_only=True
        ):
            va_correlator[vector[0]] = np.sum(
                spins[shift] * spins, axis=1,  # sum over vector components
            ).mean(
                axis=0,  # average over volume
            )
        # Make connected
        va_correlator -= self._magnetisation_sq(spins) / self.lattice.volume ** 2

        return va_correlator

    def _two_point_correlator(self, ensemble):
        """Calculates the two point connected correlation function for an ensemble of
        field configurations."""
        _, _, *extra_dims, _ = ensemble.shape
        correlator = np.empty((*self.lattice.dimensions, *extra_dims, 1))
        for vector, shift in self.lattice.two_point_iterator():
            correlator[vector] = (
                np.sum(
                    ensemble[shift] * ensemble, axis=1,  # sum over vector components
                ).mean(
                    axis=-1,  # average over ensemble
                    keepdims=True,  # keep ensemble dimension
                )
                - np.sum(
                    ensemble[shift].mean(
                        axis=-1, keepdims=True,  # average over ensemble
                    )
                    * ensemble.mean(axis=-1, keepdims=True),  # average over ensemble
                    axis=1,  # sum over vector components
                )
            ).mean(
                axis=0  # average over volume
            )
        return correlator

    def _spherical_triangle_area(self, a, b, c):
        """Helper function which calculates the surface area of a unit sphere enclosed
        by geodesics between three points on the surface. The parameters are the unit
        vectors corresponding the three points on the unit sphere.
        """
        return 2 * np.arctan2(  # arctan2 since output needs to be (-2pi, 2pi)
            np.sum(a * np.cross(b, c, axis=0), axis=0),  # numerator
            1 + np.sum(a * b, axis=0) + np.sum(b * c, axis=0) + np.sum(c * a, axis=0),  # denominator
        )

    def _topological_charge(self, spins):
        """Calculates the topological charge of a configuration or ensemble of
        Heisenberg spins."""
        _, _, *extra_dims = spins.shape
        charge = np.zeros(extra_dims)

        for x0 in range(self.lattice.volume):
            # Four points on the lattice forming a square
            x1, x3 = self.shift[x0]
            x2 = self.shift[x1, 1]

            charge += self._spherical_triangle_area(*list(spins[[x0, x1, x2]]))
            charge += self._spherical_triangle_area(*list(spins[[x0, x2, x3]]))

        return charge / (4 * pi)

    @bootstrapped
    def _boot_hamiltonian(self, ensemble):
        """Calculates the Hamiltonian for a bootstrap sample of ensembles."""
        return self._hamiltonian(ensemble)

    @bootstrapped
    def _boot_magnetisation_sq(self, ensemble):
        """Calculates the square of the magnetisation for a bootstrap sample of ensembles."""
        return self._magnetisation_sq(ensemble)

    @bootstrapped
    def _boot_two_point_correlator(self, ensemble):
        """Calculates the two point connected correlation function for a bootstrap sample
        of ensembles."""
        return self._two_point_correlator(ensemble)

    @bootstrapped
    def _boot_topological_charge(self, spins):
        """Calculates the topological charge of a bootstrap sample of ensembles of
        of Heisenberg spins."""
        return self._topological_charge(spins)

    @property
    def spins(self):
        """The configuration or ensemble of N-dimensional spin vectors (alias of coords).
        numpy.ndarray, dimensions (lattice_volume, N, *, ensemble_size)"""
        return self.coords

    @unit_norm(dim=1)
    @spins.setter
    def spins(self, new):
        """Updates the spin configuration or ensemble (by updating coords), also checking
        that the spin vectors have unit norm."""
        self.coords = new  # calls coords.__set__

    @property
    def hamiltonian(self):
        """The spin Hamiltonian for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._hamiltonian(self.spins)
    
    @property
    def action(self):
        """The O(N) action for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self.beta * self._hamiltonian(self.spins) + self._action_shift

    @property
    def magnetisation_sq(self):
        """The squared magnetisation for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._magnetisation_sq(self.spins)

    @property
    def vol_avg_two_point_correlator(self):
        """The volume-averaged two point connected correlation function.
        numpy.ndarray, dimensions (*lattice_dimensions, *, ensemble_size)"""
        return self._vol_avg_two_point_correlator(self.spins)

    @property
    @requires_topology
    def topological_charge(self):
        """Topological charge of a configuration or ensemble of Heisenberg spins,
        according to the geometrical definition given in REF
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._topological_charge(self.spins)

    @property
    @requires_ensemble
    def two_point_correlator(self):
        """Two point connected correlation function, where the ensemble average
        is taken before the volume average.
        numpy.ndarray, dimensions (*lattice_dimensions, *, 1)"""
        return self._two_point_correlator(self.spins)

    @property
    @requires_ensemble
    def boot_hamiltonian(self):
        """The spin Hamiltonian for a bootstrap sample of ensembles.
        numpy.ndarray, dimensions(*, bootstrap_sample_size, ensemble_size)"""
        # NOTE: Not sure why the instance isn't automatically passed to the
        # __call__ method of @bootstrapped!!!
        return self._boot_hamiltonian(self, self.spins)
    
    @property
    @requires_ensemble
    def boot_action(self):
        """The O(N) action for a bootstrap sample of ensembles.
        numpy.ndarray, dimensions(*, bootstrap_sample_size, ensemble_size)"""
        # NOTE: this is a wasteful additional calculation
        return self.beta * self._boot_hamiltonian(self, self.spins) + self._action_shift

    @property
    @requires_ensemble
    def boot_magnetisation_sq(self):
        """The squared magnetisation for a bootstrap sample of ensembles.
        numpy.ndarray, dimensions(*, bootstrap_sample_size, ensemble_size)"""
        return self._boot_magnetisation_sq(self, self.spins)

    @property
    @requires_ensemble
    def boot_two_point_correlator(self):
        """Two point connected correlation function for a bootstrap sample of
        ensembles, where the ensemble average is taken before the volume average.
        numpy.ndarray, dimensions (*lattice_dimensions, *, bootstrap_sample_size, 1)"""
        return self._boot_two_point_correlator(self, self.spins)

    @property
    @requires_ensemble
    @requires_topology
    def boot_topological_charge(self):
        """Topological charge of a bootstrap sample of ensembles of Heisenberg spins.
        numpy.ndarray, dimensions (*, bootstrap_sample_size, ensemble_size)"""
        return self._boot_topological_charge(self, self.spins)
