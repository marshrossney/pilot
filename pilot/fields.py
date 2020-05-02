import numpy as np
from random import random, randint
from math import pi, exp
import pandas as pd

import scipy.stats
from pilot.distributions import SphericalUniformDist
from pilot.utils import (
    spher_to_eucl,
    bootstrapped,
    unit_norm,
    method_property,
    requires_true,
)


class HamiltonianMismatchError(Exception):
    pass


class _valid_field:
    def __init__(self, setter):
        self.setter = setter

    def __call__(self, instance, array_in):
        # Check that 0th dimension of input data matches the number of lattice sites
        assert (
            array_in.shape[0] == instance.lattice.volume
        ), f"Size of coordinates array at dimension 0: {array_in.shape[0]} does not match volume of lattice: {instance.lattice.volume}"
        # Check that the input data has at least the minimum number of dimensions
        n_dims = len(array_in.shape)
        assert (
            n_dims >= instance.minimum_dimensions
        ), f"Invalid number of dimensions in coordinate array. Expected {instance.minimum_dimensions} or more, but found {n_dims}."
        # If missing, add ensemble dimension for convenience
        if n_dims == instance.minimum_dimensions:
            array_in = np.expand_dims(array_in, axis=-1)

        self.setter(instance, array_in)
        return


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
        # TODO cached property called only when needed?
        self.neighbours = self.lattice.get_neighbours()

        self.has_topology = False

    def __str__(self):
        header = f"Field: {type(self).__name__}"
        line = "".join(["-" for char in header])
        out = "\n" + header + "\n" + line
        for prop in self.summary_properties:
            value = getattr(self, prop)
            label = prop.replace("_", " ")
            out += f"\n{label}: {value}"
        return out

    @property
    def coords(self):
        """The set of coordinates which define the configuration or ensemble.
        numpy.ndarray, dimensions (lattice_volume, N, *, ensemble_size)"""
        return self._coords

    @coords.setter
    @_valid_field
    def coords(self, new):
        """Setter for coords which performs some basic checks (see valid_field)."""
        self._coords = new

    @property
    def is_ensemble(self):
        """Bool indicating whether the coordinates have an ensemble dimension with a
        size greater than one."""
        if len(self.coords.squeeze().shape) >= self.minimum_dimensions + 1:
            return True
        return False

    @property
    def is_single(self):
        """Bool indicating whether the coordinates array is just a single field."""
        return not self.is_ensemble

    @property
    def has_topology(self):
        """Bool or indicating whether or not topological properties can be calculated.
        May be False due to lack of implementation or because the topology is trivial."""
        return self._has_topology

    @has_topology.setter
    def has_topology(self, has):
        """Setter for has_topology."""
        self._has_topology = has


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

    generators = {"uniform": SphericalUniformDist}
    minimum_dimensions = 2
    summary_properties = [
        "euclidean_dimension",
        "beta",
    ]

    def __init__(self, input_coords, lattice, beta=1.0):
        super().__init__(input_coords, lattice, beta=beta)

        self.spins = self.coords
        self.euclidean_dimension = self.spins.shape[1]

        if self.euclidean_dimension == 3:
            self.has_topology = True

    @classmethod
    def from_spherical(cls, input_coords, lattice, beta=1.0):
        """Constructs an instance of this class where the input coordinates are in the
        spherical representation."""
        return cls(spher_to_eucl(input_coords), lattice, beta=beta)

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
        if input_spherical:
            return cls.from_spherical(
                input_coords, template.lattice, beta=template.beta
            )
        return cls(input_coords, template.lattice, beta=template.beta)

    @classmethod
    def from_random(cls, lattice, *, N, ensemble_size=1, **theory_kwargs):
        # TODO generalise
        input_coords = cls.generators["uniform"](dim=N).rvs(
            size=(lattice.volume, ensemble_size)
        )
        return cls(input_coords, lattice, **theory_kwargs)

    def _hamiltonian(self, spins):
        """Calculates the Hamiltonian for a spin system or ensemble."""
        return -self.beta * np.sum(
            spins[self.shift] * np.expand_dims(spins, axis=1),
            axis=2,  # sum over vector components
        ).sum(
            axis=0,  # sum over dimensions
        ).sum(
            axis=0,  # sum over volume
        )

    def _magnetisation_sq(self, spins):
        """Calculates the square of the magnetisation for a spin system or ensemble."""
        mag = spins.mean(axis=0)  # volume average
        return np.sum(mag ** 2, axis=0)  # sum over vector components

    def _vol_avg_two_point_correlator(self, spins):
        """Calculates the volume-averaged two point connected correlation function for an
        ensemble of field configurations."""
        _, _, *extra_dims = spins.shape
        va_correlator = np.empty((*self.lattice.dimensions, *extra_dims))

        # Disconnected part
        for vector, shift in self.lattice.two_point_iterator():
            va_correlator[vector] = np.sum(
                spins[shift] * spins, axis=1,  # sum over vector components
            ).mean(
                axis=0,  # average over volume
            )
        # Make connected
        va_correlator -= self._magnetisation_sq(spins)

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
        numerator = np.sum(
            a * np.cross(b, c, axis=0), axis=0
        )  # keep dim so we always output an array
        denominator = (
            1 + np.sum(a * b, axis=0) + np.sum(b * c, axis=0) + np.sum(c * a, axis=0)
        )
        return 2 * np.arctan(numerator / denominator)

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
        return self._hamiltonian(ensemble)

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
        # TODO: check and adjust dimensions here
        self.coords = new

    @method_property
    def hamiltonian(self):
        """The Hamiltonian for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._hamiltonian(self.spins)

    @method_property
    def magnetisation_sq(self):
        """The squared magnetisation for each configuration in the ensemble.
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._magnetisation_sq(self.spins)

    @method_property
    def vol_avg_two_point_correlator(self):
        """The volume-averaged two point connected correlation function.
        numpy.ndarray, dimensions (*lattice_dimensions, *, ensemble_size)"""
        return self._vol_avg_two_point_correlator(self.spins)

    @method_property
    @requires_true("has_topology")
    def topological_charge(self):
        """Topological charge of a configuration or ensemble of Heisenberg spins,
        according to the geometrical definition given in REF
        numpy.ndarray, dimensions (*, ensemble_size)"""
        return self._topological_charge(self.spins)

    @method_property
    @requires_true("is_ensemble")
    def two_point_correlator(self):
        """Two point connected correlation function, where the ensemble average
        is taken before the volume average.
        numpy.ndarray, dimensions (*lattice_dimensions, *, 1)"""
        return self._two_point_correlator(self.spins)

    @method_property
    @requires_true("is_ensemble")
    def boot_hamiltonian(self):
        """The Hamiltonian for a bootstrap sample of ensembles.
        numpy.ndarray, dimensions(*, bootstrap_sample_size, ensemble_size)"""
        # NOTE: Not sure why the instance isn't automatically passed to the
        # __call__ method of @bootstrapped!!!
        return self._boot_hamiltonian(self, self.spins)

    @method_property
    @requires_true("is_ensemble")
    def boot_magnetisation_sq(self):
        """The squared magnetisation for a bootstrap sample of ensembles.
        numpy.ndarray, dimensions(*, bootstrap_sample_size, ensemble_size)"""
        return self._boot_magnetisation_sq(self, self.spins)

    @method_property
    @requires_true("is_ensemble")
    def boot_two_point_correlator(self):
        """Two point connected correlation function for a bootstrap sample of
        ensembles, where the ensemble average is taken before the volume average.
        numpy.ndarray, dimensions (*lattice_dimensions, *, bootstrap_sample_size, 1)"""
        return self._boot_two_point_correlator(self, self.spins)

    @method_property
    @requires_true("is_ensemble", "has_topology")
    def boot_topological_charge(self):
        """Topological charge of a bootstrap sample of ensembles of Heisenberg spins.
        numpy.ndarray, dimensions (*, bootstrap_sample_size, ensemble_size)"""
        return self._boot_topological_charge(self, self.spins)

    @requires_true("is_single")
    def metropolis_update(self, sweeps=1, debug=False):
        """Perform a sequence of local updates according to the Metropolis algorithm.
        
        Parameters
        ----------
        sweeps: int
            The number of proposals generated will be (lattice.volume * sweeps).
        debug: bool
            If True, check that the result of local updates matches the re-computed
            Hamiltonian. TODO: replace with some clever logging thing.
        
        Updates
        -------
        The following attributes will be updated:
            - spins
            - hamiltonian

        Returns
        -------
        spins: numpy.ndarray
            The field configuration resulting from the sequence of local updates.
            Dimensions (lattice_volume, N)
        n_accept: int
            The number of proposals that were accepted.

        Notes
        -----
        A local update comprises the rotation of a single spin by an element of O(N).
        These rotation matrices are generated batch-wise using scipy.stats.ortho_group.
        """
        # TODO check that we have a single config, not an ensemble
        n_proposals = self.lattice.volume * sweeps
        n_accept = 0

        # Make local copies of spins and Hamiltonian
        spins = np.squeeze(self.spins.copy())
        hamiltonian = float(self.hamiltonian)

        # Generate a batch of random rotation matrices
        rotations = scipy.stats.ortho_group.rvs(
            dim=self.euclidean_dimension, size=n_proposals
        )

        for i in range(n_proposals):
            rotation = rotations[i]
            site = randint(0, self.lattice.volume - 1)

            current = spins[site]
            proposal = np.dot(rotation, current)

            delta_hamil = -self.beta * np.dot(
                proposal - current, spins[self.neighbours[site]].sum(axis=0)
            )

            if random() < exp(-delta_hamil):
                spins[site] = proposal
                hamiltonian += delta_hamil
                n_accept += 1

        # Update object copy of spins
        self.spins = np.expand_dims(spins, axis=-1)

        if debug:
            # Recalculate Hamiltonian using spins and check it matches local version
            error = float(self.hamiltonian - hamiltonian)
            if abs(error) > 1e-6:
                raise HamiltonianMismatchError(
                    f"Disagreement between full calculation {hamiltonian + error} and result of local updates {hamiltonian}"
                )

        return spins, n_accept