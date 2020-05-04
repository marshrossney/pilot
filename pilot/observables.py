"""
observables.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import ceil
from itertools import product

from pilot.utils import cached_property, NotDefinedForField


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


def autocorrelation(chain):
    chain -= chain.mean(axis=-1, keepdims=True)  # expect ensemble dimension at -1
    auto = correlate(chain, chain, mode="same")
    t0 = auto.shape[-1] // 2  # this is true for mode="same"
    return auto[..., t0:] / auto[..., [t0]]  # normalise and take +ve shifts


def optimal_window(integrated, mult=2.0, eps=1e-6):

    # Exponential autocorrelation
    with np.errstate(invalid="ignore", divide="ignore"):
        exponential = np.clip(
            np.nan_to_num(mult / np.log((2 * integrated + 1) / (2 * integrated - 1))),
            a_min=eps,
            a_max=None,
        )

    # Infer ensemble size, assuming correlation mode was 'same'!!!
    n_t = integrated.shape[-1]
    ensemble_size = n_t * 2

    # Window func, we want the minimum
    t_sep = np.arange(1, n_t + 1)
    window_func = np.exp(-t_sep / exponential) - exponential / np.sqrt(
        t_sep * ensemble_size
    )

    return np.argmax((window_func[..., 1:] < 0), axis=-1)


class Observables:
    tables = [
        "spin_observables",
        "ising_observables",
        "zero_momentum_correlator",
        "effective_pole_mass",
        "two_point_correlator",
        "topological_observables",
        "two_point_correlator_integrated_autocorrelation",
    ]
    figures = [
        "zero_momentum_correlator",
        "effective_pole_mass",
        "two_point_correlator",
        "two_point_correlator_series",
        "two_point_correlator_autocorrelation",
        "topological_charge_series",
        "topological_charge_autocorrelation",
    ]

    def __init__(self, ensemble, bootstrap=True):
        self.ensemble = ensemble
        self.bootstrap = bootstrap

        self.volume = ensemble.lattice.volume

    def __str__(self):
        out = "\n"
        for table in self.tables:
            try:
                df = getattr(self, "table_" + table).to_string()
                header = table.replace("_", " ")
                line = "".join(["-" for char in table])
                out += f"{header}\n{line}\n{df}\n\n"
            except NotDefinedForField:
                pass

        return out

    @cached_property
    def _two_point_correlator_series(self):
        return self.ensemble.vol_avg_two_point_correlator

    @cached_property
    def _topological_charge_series(self):
        return self.ensemble.topological_charge

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

    @cached_property
    def _auto_two_point_correlator(self):
        return autocorrelation(self.ensemble.vol_avg_two_point_correlator)

    @cached_property
    def _auto_topological_charge(self):
        return autocorrelation(self.ensemble.topological_charge)

    @cached_property
    def _iauto_two_point_correlator(self):
        return np.cumsum(self._auto_two_point_correlator, axis=-1) - 0.5

    @cached_property
    def _iauto_topological_charge(self):
        return np.cumsum(self._auto_topological_charge) - 0.5

    @cached_property
    def _optimal_window_two_point_correlator(self):
        return optimal_window(self._iauto_two_point_correlator)

    @cached_property
    def _optimal_window_topological_charge(self):
        return optimal_window(self._iauto_topological_charge)

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

    @property
    def table_spin_observables(self):
        keys = [
            "energy_density",
            "magnetisation_sq",
            "magnetic_susceptibility",
            "heat_capacity",
        ]
        return pd.DataFrame(
            [getattr(self, key) for key in keys],
            index=[key.replace("_", " ") for key in keys],
            columns=["value", "error"],
        )

    @property
    def table_ising_observables(self):
        keys = ["ising_energy", "susceptibility"]
        return pd.DataFrame(
            [getattr(self, key) for key in keys],
            index=[key.replace("_", " ") for key in keys],
            columns=["value", "error"],
        )

    @property
    def table_zero_momentum_correlator(self):
        return pd.DataFrame(self.zero_momentum_correlator, index=["value", "error"]).T

    @property
    def table_effective_pole_mass(self):
        return pd.DataFrame(self.effective_pole_mass, index=["value", "error"]).T

    @property
    def table_two_point_correlator(self):
        value, error = self.two_point_correlator
        return pd.concat(
            [pd.DataFrame(value), pd.DataFrame(error)], keys=["value", "error"],
        )

    @property
    def table_topological_observables(self):
        keys = ["topological_charge", "topological_susceptibility"]
        return pd.DataFrame(
            [getattr(self, key) for key in keys],
            index=[key.replace("_", " ") for key in keys],
            columns=["value", "error"],
        )

    @property
    def table_two_point_correlator_integrated_autocorrelation(self):
        shape_out = self._iauto_two_point_correlator.shape[:-1]
        w_opt = self._optimal_window_two_point_correlator
        values_out = np.empty(shape_out)
        for tup in product(*[range(size) for size in shape_out]):
            index = tup + (w_opt[tup],)
            values_out[tup] = self._iauto_two_point_correlator[index]
        return pd.DataFrame(values_out)

    @property
    def plot_zero_momentum_correlator(self):
        df = self.table_zero_momentum_correlator
        return df["value"].plot(
            x=df.index.values, yerr=df["error"], title="zero momentum correlator"
        )

    @property
    def plot_effective_pole_mass(self):
        df = self.table_effective_pole_mass
        return df["value"].plot(
            x=df.index.values, yerr=df["error"], title="effective pole mass"
        )

    @property
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

    @property
    def plot_two_point_correlator_series(self):
        fig, ax = plt.subplots(1)
        ax.set_title("Two point correlator series$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$G(x; t)$")
        for i in range(self._two_point_correlator_series.shape[0]):
            ax.plot(
                self._two_point_correlator_series[i, :],
                linewidth=1,
                label=f"$x =$ ({i}, {i})",
            )
        ax.legend()
        fig.tight_layout()
        return fig

    @property
    def plot_topological_charge_series(self):
        fig, ax = plt.subplots(1)
        ax.set_title("Topological charge series$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$Q(t)$")
        ax.plot(self._topological_charge_series, linewidth=1)
        fig.tight_layout()
        return fig

    @property
    def plot_two_point_correlator_autocorrelation(self):
        auto_to_plot = self._auto_two_point_correlator
        integrated_to_plot = self._iauto_two_point_correlator
        lines_to_plot = self._optimal_window_two_point_correlator
        cut = max(10, 2 * np.max(lines_to_plot))

        fig, ax1 = plt.subplots(1)
        ax2 = ax1.twinx()
        ax1.set_title("$G(x; t)$ autocorrelation")
        ax1.set_xlabel("$\delta t$")
        ax1.set_ylabel("$\Gamma_G(\delta t)$")
        ax2.set_ylabel("$\sum \Gamma_G(\delta t)$")

        for i in range(auto_to_plot.shape[0]):
            color = next(ax1._get_lines.prop_cycler)["color"]
            ax1.plot(auto_to_plot[i, :cut], linestyle=":", color=color)
            ax2.plot(
                integrated_to_plot[i, :cut],
                linestyle="-",
                color=color,
                label=f"$x =$ ({i},{i})",
            )
            ax1.axvline(lines_to_plot[i], linestyle="--", color=color)

        ax1.set_xlim(left=0)
        ax1.set_ylim(top=1)
        ax2.set_ylim(bottom=0.5)

        ax2.legend()
        fig.tight_layout()
        return fig

    @property
    def plot_topological_charge_autocorrelation(self):
        auto_to_plot = self._auto_topological_charge
        integrated_to_plot = self._iauto_topological_charge
        line = self._optimal_window_topological_charge
        cut = max(10, 2 * line)

        fig, ax1 = plt.subplots(1)
        ax2 = ax1.twinx()
        ax1.set_title("$Q$ autocorrelation")
        ax1.set_xlabel("$\delta t$")
        ax1.set_ylabel("$\Gamma_Q(\delta t)$")
        ax2.set_ylabel("$\sum \Gamma_Q(\delta t)$")

        ax1.plot(auto_to_plot[:cut], linestyle=":")
        ax2.plot(integrated_to_plot[:cut], linestyle="-")
        ax1.axvline(line, linestyle="--")
        ax2.annotate(
            fr"$\tau_Q = ${integrated_to_plot[line]:.3g}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )

        ax1.set_xlim(left=0)
        ax1.set_ylim(top=1)
        ax2.set_ylim(bottom=0.5)

        fig.tight_layout()
        return fig
