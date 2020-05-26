import numpy as np
from tqdm import tqdm
import math as m
from random import random, randint
from scipy.stats import ortho_group

from pilot.utils import string_summary


class ClassicalSpinSampler:

    summary_properties = [
        "sample_size",
        "sample_interval",
        "thermalisation",
        "algorithm",
        "delta",
        "acceptance_fraction",
    ]

    def __init__(self, field):

        self.field = field
        self.beta = field.beta
        self.volume = field.lattice.volume
        self.neighbours = field.lattice.get_neighbours()

        # Make local copies of spins and action
        self.spins = np.squeeze(field.spins.copy())
        self.hamiltonian = float(field.hamiltonian)

        # For the benefit of the report
        self.algorithm = "metropolis"

    def __str__(self):
        return string_summary(self, "Algorithm")

    def reset(self, new_field):
        self.field = new_field
        self.spins = np.squeeze(new_field.spins.copy())
        self.hamiltonian = float(new_field.hamiltonian)
        return

    def _metropolis_condition(self, site, current, proposal):
        delta_hamil = -np.dot(
            proposal - current, self.spins[self.neighbours[site]].sum(axis=0)
        )
        if random() < m.exp(-self.beta * delta_hamil):
            self.spins[site] = proposal
            self.hamiltonian += delta_hamil
            return 1
        return 0
    
    def update(self, site):
        current = self.spins[site]
        proposal = np.dot(
            scipy.stats.ortho_group.rvs(dim=self.field.euclidean_dimension), current
        )
        return self._metropolis_condition(site, current, proposal)
    
    def __call__(self, sample_size, sample_interval=1, thermalisation=1):
        # Thermalise
        pbar = tqdm(range(thermalisation), desc="thermalisation")
        for sweep in pbar:
            for t in range(self.volume):
                site = randint(0, self.volume - 1)
                _ = self.update(site)

        sample = np.empty(
            (self.volume, self.field.euclidean_dimension, sample_size)
        )
        n_accepted = 0
        pbar = tqdm(range(sample_size), desc="Sampling batch")
        for batch in pbar:  # loop over configurations in output sample
            for t in range(self.volume * sample_interval):
                site = randint(0, self.volume - 1)
                n_accepted += self.update(site)

            sample[:, :, batch] = self.spins

        # Update attributes for benefit of report.
        # TODO: this really needs improving
        self.acceptance_fraction = n_accepted / (
            sample_size * sample_interval * self.volume
        )
        self.sample_size = sample_size
        self.sample_interval = sample_interval
        self.thermalisation = thermalisation

        # TODO: add back in a check that the action/hamiltonian matches

        return sample

        
class XYSampler(ClassicalSpinSampler):

    def __init__(self, field, algorithm="metropolis", delta=1):
        super().__init__(field)
        self.algorithm = algorithm

        if algorithm == "metropolis":
            self.update = self.metropolis_update
            self.delta = delta
        elif algorithm == "heatbath":
            self.update = self.heatbath_update
            self.delta = "N/A"

    def metropolis_update(self, site):
        current = self.spins[site]

        # If we generate cos_alpha, need a second random number for sign of sin_alpha
        # Hence, just generate alpha
        alpha = (random() - 0.5) * 2 * m.pi * self.delta
        sin_alpha = m.sin(alpha)
        cos_alpha = m.cos(alpha)
        proposal = np.dot(
            np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]]), current
        )
        return self._metropolis_condition(site, current, proposal)

    
class HeisenbergSampler(XYSampler):
    
    @staticmethod
    def _matrix_representation(spin):
        # ref_vector in form of a matrix acting on (0, 0, 1)
        # 3-dimensional spins only
        cos_theta = spin[2]
        sin_theta = m.sqrt(1 - cos_theta ** 2)
        cos_phi = spin[0] / sin_theta
        sin_phi = spin[1] / sin_theta
        return np.array(
            [
                [cos_theta * cos_phi, -sin_phi, sin_theta * cos_phi],
                [cos_theta * sin_phi, cos_phi, sin_theta * sin_phi],
                [-sin_theta, 0, cos_theta],
            ]
        )

    def metropolis_update(self, site):
        current = self.spins[site]

        alpha = random() * 2 * m.pi
        cos_gamma = 1 - random() * 2 * self.delta
        sin_gamma = m.sqrt(1 - cos_gamma ** 2)
        proposal = np.dot(
            self._matrix_representation(current),
            np.array([sin_gamma * m.cos(alpha), sin_gamma * m.sin(alpha), cos_gamma,]).T,
        )

        return self._metropolis_condition(site, current, proposal)

    def heatbath_update(self, site):
        current = self.spins[site]

        local_field = self.spins[self.neighbours[site]].sum(axis=0)
        local_field_magnitude = np.linalg.norm(local_field)
        coupling = self.beta * local_field_magnitude

        # Representation of local field unit vector as rotation matrix acting on (0, 0, 1)
        rotation = self._matrix_representation(local_field / local_field_magnitude)

        # Generate proposal for polar angle
        x = random()
        cos_gamma = m.log(m.exp(coupling) * (1 - x) + m.exp(-coupling) * x) / coupling
        sin_gamma = m.sqrt(1 - cos_gamma ** 2)
        
        alpha = (random() - 0.5) * 2 * m.pi
        sin_alpha = m.sin(alpha)
        cos_alpha = m.cos(alpha)


        self.spins[site] = np.dot(
            rotation,
            np.array([sin_gamma * cos_alpha, sin_gamma * sin_alpha, cos_gamma]),
        )
        return 1

