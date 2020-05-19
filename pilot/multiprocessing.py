import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import islice
from math import ceil

USE_MULTIPROCESSING = True

class Multiprocessing:
    def __init__(self, func, generator):
        self.func = func
        self.generator = generator

        self.n_iters = sum(1 for _ in generator())
        self.n_cores = mp.cpu_count()
        if not USE_MULTIPROCESSING:
            self.n_cores = 1
        self.max_chunk = ceil(self.n_iters / self.n_cores)

    def target(self, k, output_dict):
        # NOTE: separate generator object for each proc
        generator_k = islice(self.generator(), k * self.max_chunk, min((k + 1) * self.max_chunk, self.n_iters))
        i_glob = k * self.max_chunk
        for i, args in enumerate(generator_k):
            output_dict[i_glob + i] = self.func(args)
        return

    def __call__(self):
        manager = mp.Manager()
        output_dict = manager.dict()

        procs = []
        for k in range(self.n_cores):
            p = mp.Process(target=self.target, args=(k, output_dict,),)
            procs.append(p)
            p.start()

        # Kill the zombies
        for p in procs:
            p.join()

        return output_dict



BOOTSTRAP_SAMPLE_SIZE = 1

class bootstrapped:
    """
    For field observables: expects an ensemble dimension as the last dimension!!!
    """

    def __init__(self, func):
        self._func = func
        self.__doc__ = func.__doc__

        self._n_cores = 1  # mp.cpu_count()
        self._chunk_size = ceil(BOOTSTRAP_SAMPLE_SIZE / self._n_cores)

    def _get_data_size(self, args):
        sizes = []
        for arg in args:
            assert isinstance(arg, np.ndarray), "Input is not a numpy.ndarray."
            sizes.append(arg.shape[-1])  # expect ensemble to be last dimension

        assert (
            len(set(sizes)) == 1
        ), "Inputs have different sizes at the final dimension."
        return sizes[0]

    def resample_func(self, i, sample, data_size, instance, args, kwargs):
        state = np.random.RandomState(i)
        k = i * self._chunk_size

        for j in range(self._chunk_size):
            boot_index = state.randint(0, data_size, size=data_size)

            resampled_args = []
            for arg in args:
                resampled_args.append(arg[..., boot_index])

            sample[j + k] = self._func(instance, *resampled_args, **kwargs)
        return

    def __call__(self, instance, *args, **kwargs):
        data_size = self._get_data_size(args)  # ensemble size

        manager = mp.Manager()
        sample = manager.dict()
        print(f"Bootstrapping {self._func.__name__}")

        procs = []
        for i in range(self._n_cores):
            p = mp.Process(
                target=self.resample_func,
                args=(i, sample, data_size, instance, args, kwargs),
            )
            procs.append(p)
            p.start()

        # Kill the zombies
        for p in procs:
            p.join()

        return np.stack(
            sample.values(), axis=-2
        )  # bootstrap dimension is the SECOND last
