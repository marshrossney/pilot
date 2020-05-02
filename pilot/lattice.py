import numpy as np
from itertools import product


class ShiftMismatchError(Exception):
    pass


class Lattice2D:
    summary_properties = [
        "dimensions",
    ]

    def __init__(self, dimensions: (int, tuple)):

        if isinstance(dimensions, int):
            self.dimensions = (dimensions, dimensions)
        else:
            self.dimensions = dimensions
        self.volume = self.dimensions[0] * self.dimensions[1]

    def __str__(self):
        header = f"Lattice: {type(self).__name__}"
        line = "".join(["-" for char in header])
        out = "\n" + header + "\n" + line
        for prop in self.summary_properties:
            value = getattr(self, prop)
            label = prop.replace("_", " ")
            out += f"\n{label}: {value}"
        return out

    def get_shift(self, shifts: tuple = (1, 1), dims: tuple = (0, 1)):
        if len(shifts) != len(dims):
            raise ShiftMismatchError(
                "Number of shifts and number of dimensions: "
                f"{len(shifts)} and {len(dims)} do not match."
            )

        state = np.arange(self.volume).reshape(self.dimensions)

        shift_index = np.zeros((len(shifts), self.volume), dtype=np.long)

        for i, (shift, dim) in enumerate(zip(shifts, dims)):
            shift_index[i, :] = np.roll(state, shift, axis=dim).flatten()

        return shift_index.transpose()  # (volume, n_shifts)

    def get_neighbours(self):
        """Returns nearest neighbour indices: up/down/left/right."""
        return self.get_shift((1, -1, 1, -1), (0, 0, 1, 1))

    def vectors(self):
        """Iterator for all lattice vectors.

        There is no reason why I need the list comp for 2D lattice.
        """
        return product(*[range(L) for L in self.dimensions])

    def two_point_iterator(self):
        """Iterator for shifts."""
        for vector in self.vectors():
            yield vector, self.get_shift((tuple(vector),), ((0, 1),)).flatten()
