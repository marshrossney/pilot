import numpy as np
from itertools import product

from pilot.utils import string_summary


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
        return string_summar(self, "Lattice")

    def get_shift(self, shifts: tuple = (1, 1), dims: tuple = (0, 1)):
        """Unnecessarily powerful method for getting the indices in the
        lexicographic representation for shifts in the Cartesian representation.
        
        TODO: document properly.
        """
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

    def two_point_iterator(self, mode="full", pos_only=False):
        """Iterator for shifts."""
        if pos_only:
            indices = [range(L // 2 + 1) for L in self.dimensions]
        else:
            indices = [range(L) for L in self.dimensions]
        if mode == "diag":
            vectors = zip(*indices)  # truncates if dimensions not equal size
        elif mode == "one_dim":
            vectors = [(0, i) for i in indices[1]]
        else:
            vectors = product(*indices)

        for vector in vectors:
            yield self.get_shift((tuple(vector),), ((0, 1),)).flatten()

    def two_point_iterator_1d(self, dim=-1, pos_only=False):
        high = self.dimensions[dim]
        if pos_only:
            high = high // 2 + 1
        vectors = [[0,] * len(self.dimensions) for _ in range(high)]
        for i, vector in enumerate(vectors):
            vector[dim] = i
            yield self.get_shift((tuple(vector),), ((0, 1),)).flatten()
