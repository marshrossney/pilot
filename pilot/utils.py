import numpy as np
import math as m
from random import random
from functools import wraps

# TODO: this should be specified in the config file, but I don't want to import from config
# here since it reduces flexibility; cannot then run any module as a script which imports
# utils without specifying a config file.
BOOTSTRAP_SAMPLE_SIZE = 100


class NotDefinedForField(Exception):
    pass


def string_summary(obj, title):
    header = f"{title}: {type(obj).__name__}"
    line = "".join(["-" for char in header])
    out = "\n" + header + "\n" + line
    for prop in obj.summary_properties:
        value = getattr(obj, prop)
        label = prop.replace("_", " ")
        out += f"\n{label}: {value}"
    return out


class bootstrapped:
    """
    For field observables: expects an ensemble dimension as the last dimension!!!
    """

    def __init__(self, func):
        self._func = func
        self._n_boot = BOOTSTRAP_SAMPLE_SIZE
        self.__doc__ = func.__doc__

    def __call__(self, instance, *args, **kwargs):
        # This version prevents potentially huge memory allocations
        # by looping over the bootstrap samples.
        # TODO: automatically select version based on array size
        data_size = self._get_data_size(args)  # ensemble size

        sample = []
        for _ in range(self._n_boot):
            boot_index = np.random.randint(0, data_size, size=data_size)
            resampled_args = []

            for arg in args:
                resampled_args.append(arg[..., boot_index])

            sample.append(self._func(instance, *resampled_args, **kwargs))

        return np.stack(sample, axis=-2)  # bootstrap dimension is the SECOND last

    def __call__alt(self, instance, *args, **kwargs):
        data_size = self._get_data_size(args)  # ensemble size

        boot_index = np.random.randint(0, data_size, size=(self._n_boot, data_size))
        resampled_args = []
        for arg in args:
            resampled_args.append(arg[..., boot_index])

        return self._func(instance, *resampled_args, **kwargs)

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
        self._dim = dim
        self._atol = atol

    def __call__(self, setter):
        @wraps(setter)
        def wrapper(instance, array_in):
            if not np.allclose(
                np.linalg.norm(array_in, axis=self._dim), 1, atol=self._atol
            ):
                raise self.UnitNormError(
                    f"Array contains elements with a norm along dimension {self.dim} that deviates from unity by more than {self.atol}."
                )

            setter(instance, array_in)


class cached_property:
    """Descriptor which evalutes a function instance.propert() and caches the
    result. Subsequently, instance.property points to this cached value rather
    than the function which calculates it.
    """

    def __init__(self, func):
        self._func = func
        self._name = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        attr = self._func(instance)

        # Cache the value, override instance.property
        setattr(instance, self._name, attr)

        # TODO: make setting illegal

        return attr


class requires:
    """Base class for decorators which will check for requirements and throw
    a custom error."""

    attributes = None
    exception = AttributeError
    message = "oops"

    def __init__(self, func):
        self._func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __call__(self, instance, *args, **kwargs):
        for attr in self.attributes:
            if not getattr(instance, attr):
                raise self.exception(self.message)

        return self._func(instance, *args, **kwargs)
