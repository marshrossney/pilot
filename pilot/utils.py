import numpy as np
import math as m
from random import random
from functools import wraps

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
