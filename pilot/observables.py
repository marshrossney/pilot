"""
observables.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil


from pilot.utils import cached_property, requires_true


class observable:
    def __init__(self, func):
        self.func = func
        self._name = func.__name__.replace("_", " ")
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        sample = self.func(instance)

        # Take mean and standard deviation over final dimension
        value = sample.mean(axis=-1)
        error = sample.std(axis=-1)

        return value, error


class table:
    def __init__(self, func):
        self._func = func
        self._name = func.__name__.replace("_", " ")
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        keys = self._func(instance)

        # [ [value1, error1], [value2, error2], ... ]
        data = [getattr(instance, key) for key in keys]

        nkeys = len(keys)
        dshape = len(data[0][0].shape)

        if nkeys > 1:  # assume all scalar values
            return pd.DataFrame(
                data,
                index=[key.replace("_", " ") for key in keys],
                columns=["value", "error"],
            )

        if dshape == 1:
            return pd.DataFrame(data[0], index=["value", "error"]).T

        elif dshape == 2:
            # [ value1, error1, value2, error2, ... ]
            data = [item for nested in data for item in nested]
            keys = ["value", "error"]
            return pd.concat([pd.DataFrame(array) for array in data], keys=keys)


class Observables:
    tables = [
        "spin_observables",
        "ising_observables",
        "zero_momentum_correlator",
        "effective_pole_mass",
        "two_point_correlator",
        "topological_observables",
    ]
    figures = [
        "zero_momentum_correlator",
        "effective_pole_mass",
        "two_point_correlator",
    ]

    def __init__(self, ensemble, bootstrap=True):
        self.ensemble = ensemble
        self.bootstrap = bootstrap

        self.volume = ensemble.lattice.volume

        self.has_topology = ensemble.has_topology
        # TODO similar with spin / thermal

    def __str__(self):
        out = "\n"
        for table in self.tables:
            try:
                df = getattr(self, "table_" + table).to_string()
                header = table.replace("_", " ")
                line = "".join(["-" for char in table])
                out += f"{header}\n{line}\n{df}\n\n"
            except AttributeError:  # TODO Should make this more specific
                pass

        return out

    @cached_property
    def _hamiltonian(self):
        return self.ensemble.boot_hamiltonian

    @cached_property
    def _magnetisation_sq(self):
        return self.ensemble.boot_magnetisation_sq

    @cached_property
    def _two_point_correlator(self):
        return self.ensemble.boot_two_point_correlator.squeeze(axis=-1)

    @cached_property
    def _fourier_space_correlator(self):
        # TODO
        pass

    @cached_property
    def _zero_momentum_correlator(self):
        return self._two_point_correlator.mean(axis=0)

    @cached_property
    def _topological_charge(self):
        return self.ensemble.boot_topological_charge

    @observable
    def energy_density(self):
        return self._hamiltonian.mean(axis=-1) / self.volume

    @observable
    def magnetisation_sq(self):
        return self._magnetisation_sq.mean(axis=-1)

    @observable
    def magnetic_susceptibility(self):
        return self._magnetisation_sq.var(axis=-1) / self.volume

    @observable
    def heat_capacity(self):
        return self.ensemble.beta ** 2 / self.volume * self._hamiltonian.var(axis=-1)

    @observable
    def two_point_correlator(self):
        return self._two_point_correlator

    @observable
    def ising_energy(self):
        return self._two_point_correlator[1, 0] + self._two_point_correlator[0, 1]

    @observable
    def susceptibility(self):
        return self._two_point_correlator.sum(axis=(0, 1))

    @observable
    def zero_momentum_correlator(self):
        return self._zero_momentum_correlator

    @observable
    def effective_pole_mass(self):
        inner_indices = np.arange(
            1, self._zero_momentum_correlator.shape[0] - 1, dtype=int
        )
        epm = np.arccosh(
            (
                self._zero_momentum_correlator[inner_indices - 1]
                + self._zero_momentum_correlator[inner_indices + 1]
            )
            / (2 * self._zero_momentum_correlator[inner_indices])
        )
        return epm

    @observable
    def topological_charge(self):
        return self._topological_charge.mean(axis=-1)

    @observable
    def topological_susceptibility(self):
        return self._topological_charge.var(axis=-1) / self.volume

    @table
    def table_spin_observables(self):
        return [
            "energy_density",
            "magnetisation_sq",
            "magnetic_susceptibility",
            "heat_capacity",
        ]

    @table
    def table_ising_observables(self):
        return ["ising_energy", "susceptibility"]

    @table
    def table_zero_momentum_correlator(self):
        return [
            "zero_momentum_correlator",
        ]

    @table
    def table_effective_pole_mass(self):
        return [
            "effective_pole_mass",
        ]

    @table
    def table_two_point_correlator(self):
        return [
            "two_point_correlator",
        ]

    @table
    def table_topological_observables(self):
        return ["topological_charge", "topological_susceptibility"]

    def plot_zero_momentum_correlator(self):
        df = self.table_zero_momentum_correlator
        return df["value"].plot(
            x=df.index.values, yerr=df["error"], title="zero momentum correlator"
        )

    def plot_effective_pole_mass(self):
        df = self.table_effective_pole_mass
        return df["value"].plot(
            x=df.index.values, yerr=df["error"], title="effective pole mass"
        )

    def plot_two_point_correlator(self):
        # NOTE: Annoyingly seems to sort things in alphabetical order
        iterator = self.table_two_point_correlator.groupby(level=0)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        for ax, (key, df) in zip(reversed(axes), iterator):
            img = ax.imshow(df)
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            ax.xaxis.tick_top()
            ax.set_title(key)
        fig.tight_layout()
        return fig
