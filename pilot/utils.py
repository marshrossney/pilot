import numpy as np
from functools import wraps


# TODO: set this somewhere sensible
BOOTSTRAP_SAMPLE_SIZE = 100


class NotDefinedForField(Exception):
    pass


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


class bootstrapped:
    """
    For field observables: expects an ensemble dimension as the last dimension!!!
    """

    def __init__(self, func):
        self.func = func
        self.n_boot = BOOTSTRAP_SAMPLE_SIZE
        self.__doc__ = func.__doc__

    def __call__alt(self, instance, *args, **kwargs):
        # This version can save memory
        # TODO: automatically select version based on array size
        data_size = self._get_data_size(args)  # ensemble size

        sample = []
        for _ in range(self.n_boot):
            boot_index = np.random.randint(0, data_size, size=data_size)
            resampled_args = []

            for arg in args:
                resampled_args.append(arg[..., boot_index])

            func_output = self.func(instance, *resampled_args, **kwargs)
            sample.append(func_output)

        return np.stack(sample, axis=-2)  # bootstrap dimension is the SECOND last

    def __call__(self, instance, *args, **kwargs):
        data_size = self._get_data_size(args)  # ensemble size

        boot_index = np.random.randint(0, data_size, size=(self.n_boot, data_size))
        resampled_args = []
        for arg in args:
            resampled_args.append(arg[..., boot_index])

        func_output = self.func(instance, *resampled_args, **kwargs)
        return func_output

    def _get_data_size(self, args):
        sizes = []
        for arg in args:
            assert isinstance(arg, np.ndarray), "Input is not a numpy.ndarray."
            sizes.append(arg.shape[-1])  # expect ensemble to be last dimension

        assert (
            len(set(sizes)) == 1
        ), "Inputs have different sizes at the final dimension."
        return sizes[0]


class unit_norm:
    class UnitNormError(Exception):
        pass

    def __init__(self, dim=1, atol=1e-6):
        self.dim = dim
        self.atol = atol

    def __call__(self, setter):
        @wraps(setter)
        def wrapper(instance, array_in):
            if not np.allclose(
                np.linalg.norm(array_in, axis=self.dim), 1, atol=self.atol
            ):
                raise self.UnitNormError(
                    f"Array contains elements with a norm along dimension {self.dim} that deviates from unity by more than {self.atol}."
                )

            setter(instance, array_in)


class cached_property:
    def __init__(self, func):
        self._func = func
        self._name = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        attr = self._func(instance)

        # Cache the value
        setattr(instance, self._name, attr)

        # TODO: make setting illegal

        return attr


class requires:

    attributes = None
    exception = AttributeError
    message = "oops"

    def __init__(self, func):
        self.func = func

    def __call__(self, instance, *args, **kwargs):
        for attr in self.attributes:
            if not getattr(instance, attr):
                raise self.exception(self.message)

        return self.func(instance, *args, **kwargs)
