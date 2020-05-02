import numpy as np
from tqdm import tqdm
import pandas as pd


class Metropolis:
    """
    Class which implements the Metropolis algorithm
    """

    summary_properties = [
        "sample_size",
        "thermalisation",
        "sample_interval",
    ]

    def __init__(self, field, sample_size=10000, thermalisation=1, sample_interval=1):
        self.field = field
        self.sample_size = sample_size
        self.thermalisation = thermalisation
        self.sample_interval = sample_interval

    def __str__(self):
        header = f"Algorithm: {type(self).__name__}"
        line = "".join(["-" for char in header])
        out = "\n" + header + "\n" + line
        for prop in self.summary_properties:
            value = getattr(self, prop)
            label = prop.replace("_", " ")
            out += f"\n{label}: {value}"
        return out

    def thermalise(self):
        """Allow the algorithm to thermalise by performing a number of updates without
        saving the field configurations.

        The number of update 'sweeps' is given by self.thermalisation, where each sweep
        comprises the same number of local updates as there are sites on the lattice.
        """
        for t in range(self.thermalisation):
            _ = self.field.metropolis_update()
        return

    def run(self):

        self.thermalise()

        sample = np.empty(
            (
                self.field.lattice.volume,
                self.field.euclidean_dimension,
                self.sample_size,
            )
        )
        total_proposed = (
            self.field.lattice.volume * self.sample_interval * self.sample_size
        )
        total_accepted = 0

        pbar = tqdm(range(self.sample_size), desc="Sampling batch")
        for t in pbar:

            state, accepted = self.field.metropolis_update(sweeps=self.sample_interval)
            sample[:, :, t] = state
            total_accepted += accepted

        print(rf"Accepted {total_accepted / total_proposed * 100:.3g}% of proposals.")

        return sample
