# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import warnings
import itertools
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import nevergrad.common.typing as tp
from nevergrad.optimization.utils import SequentialExecutor
from .experiments import registry as registry
from .experiments import Experiment as Experiment
from . import utils


def import_additional_module(filepath: tp.PathLike) -> None:
    """Imports an additional file at runtime

    Parameter
    ---------
    filepath: str or Path
        the file to import
    """
    filepath = Path(filepath)
    spec = importlib.util.spec_from_file_location("nevergrad.additionalimport." + filepath.with_suffix("").name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore


def save_or_append_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Saves a dataframe to a file in append mode
    """
    if path.exists():
        print("Appending to existing file")
        predf = pd.read_csv(str(path))
        df = pd.concat([predf, df], sort=False)
    df.to_csv(path, index=False)


class Moduler:
    """Provides a selector of indices based on the modulo
    moduler(number) will be true iff number = modulo * k + index with k an integer

    Parameters
    ----------
    modulo: int
        modulo for number selection
    index: int
        the congruence of the number for the moduler function to evaluate to True
    total_length: int or None
        total length of the sequence the moduler will be applied on. If provided,
        this allows to compute the length of the modulated sequence.
    """

    def __init__(self, modulo: int, index: int, total_length: tp.Optional[int] = None) -> None:
        assert modulo > 0, "Modulo must be strictly positive"
        assert index < modulo, "Index must be strictly smaller than modulo"
        self.modulo = modulo
        self.index = index
        self.total_length = total_length

    def split(self, number: int) -> tp.List["Moduler"]:
        return [Moduler(self.modulo * number, self.index + k * self.modulo, self.total_length) for k in range(number)]

    def __len__(self) -> int:
        if self.total_length is None:
            raise RuntimeError("Cannot give an expected length if total_length was not provided")
        return self.total_length // self.modulo + (self.index < self.total_length % self.modulo)

    def __call__(self, index: int) -> bool:
        return (index % self.modulo) == self.index

    def __repr__(self) -> str:
        return f"Moduler({self.index}, {self.modulo}, total_length={self.total_length})"


class BenchmarkChunk:
    """Splittable chunk of a benchmark

    Parameters
    ----------
    name: str
        Name of the benchmark
    repetitions: int
        Number of repetitions to perform on the benchmark
    seed: int
        A seed for the experiment plan (if seedable)
    cap_index: int
        index at which the experiment plan must be stopped (convenient for testing if the experiment
        plan holds 10k experiment, we can select the first cap_index=100 for instance)
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, name: str, repetitions: int = 1, seed: tp.Optional[int] = None, cap_index: tp.Optional[int] = None) -> None:
        self.name = name
        self.seed = seed
        self.cap_index = None if cap_index is None else max(1, int(cap_index))
        self._moduler: tp.Optional[Moduler] = None
        self.repetitions = repetitions
        self.summaries: tp.List[tp.Dict[str, tp.Any]] = []
        self._current_experiment: tp.Optional[Experiment] = None  # for stopping and resuming
        self._id = (
            datetime.datetime.now().strftime("%y-%m-%d_%H%M")
            + "_"
            + "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 4))
        )

    @property
    def moduler(self) -> Moduler:
        if self._moduler is None:
            total_length = sum(1 for _ in itertools.islice(registry[self.name](), 0, self.cap_index)) * self.repetitions
            self._moduler = Moduler(1, 0, total_length=total_length)
        return self._moduler

    @property
    def id(self) -> str:
        """Unique ID which can be used to print in a file for instance
        """
        return f"{self._id}_i{self.moduler.index}m{self.moduler.modulo}"

    def __iter__(self) -> tp.Iterator[tp.Tuple[int, Experiment]]:
        maker = registry[self.name]
        seeds: tp.Iterable[tp.Optional[int]] = (
            (None for _ in range(self.repetitions)) if self.seed is None else range(self.seed, self.seed + self.repetitions)
        )
        # check experiments.py to see seedable xp
        generators = [maker() if seed is None else maker(seed=seed) for seed in seeds]
        generators = [itertools.islice(g, 0, self.cap_index) for g in generators]
        # pylint: disable=not-callable
        enumerated_selection = ((k, s) for (k, s) in enumerate(itertools.chain.from_iterable(generators)) if self.moduler(k))
        return enumerated_selection

    def split(self, number: int) -> tp.List["BenchmarkChunk"]:
        """Create n BenchmarkChunk which split the experiments of the current BenchmarkChunk

        Parameters
        ----------
        number: int
            The number of sub-chunks to create

        Returns
        -------
        list
            A list of new sub-chunks
        """
        chunks = []
        for submoduler in self.moduler.split(number):
            chunk = BenchmarkChunk(name=self.name, repetitions=self.repetitions, seed=self.seed, cap_index=self.cap_index)
            chunk._moduler = submoduler
            chunk._id = self._id
            chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return f"BenchmarkChunk({self.name}, {self.repetitions}, {self.seed}) with {self.moduler}"

    def __len__(self) -> int:
        return len(self.moduler)

    def compute(self, process_function: tp.Optional[tp.Callable[["BenchmarkChunk", Experiment], None]] = None) -> utils.Selector:
        """Run all the experiments and returns the result dataframe.

        Parameters
        ----------
        print_function: tp.Callable
            a function to print at the end of each experiment (for custom logging)
        """
        for local_ind, (index, xp) in enumerate(self):
            if local_ind < len(self.summaries):
                continue  # already computed
            indstr = f"{index} ({local_ind + 1}/{len(self)} of worker)"
            print(f"Starting {indstr}: {xp}", flush=True)
            if self._current_experiment is None:
                self._current_experiment = xp
            else:  # computation was started but interrupted (eg: KeyboardInterrupt)
                if xp != self._current_experiment:
                    warnings.warn(f"Could not resume unfinished xp: {self._current_experiment}")
                    self._current_experiment = xp
                else:
                    opt = self._current_experiment._optimizer
                    if opt is not None:
                        print(f"Resuming existing experiment from iteration {opt.num_ask}.", flush=True)
            self._current_experiment.run()
            summary = self._current_experiment.get_description()
            if process_function is not None:
                process_function(self, self._current_experiment)
            self.summaries.append(summary)
            self._current_experiment = None
            print(f"Finished {indstr}", flush=True)
        return utils.Selector(data=self.summaries)


# pylint: disable=too-many-arguments
def _submit_jobs(
    experiment_name: str,
    num_workers: int = 1,
    seed: tp.Optional[int] = None,
    executor: tp.Optional[tp.ExecutorLike] = None,
    print_function: tp.Optional[tp.Callable[[Experiment], None]] = None,
    cap_index: tp.Optional[int] = None,
) -> tp.List[tp.JobLike[utils.Selector]]:
    """Submits a job for computation

    Parameters
    ----------
    experiment_name: str
        name of the experiment plan (must be registered in experiments.registry)
    num_workers: int
        number of workers onto which the jobs will be distributed
    seed: int
        a seed for the experiment plan (if seedable)
    executor: Executor-like object
        an object such as concurrent.futures.ThreadPoolExecutor for running experiments in parallel
    print_function: tp.Callable
        a function to print at the end of each experiment (for custom logging)
    cap_index: int
        index at which the experiment plan must be stopped (convenient for testing if the experiment
        plan holds 10k experiment, we can select the first cap_index=100 for instance)

    Returns
    -------
    list
        A list of jobs corresponding to each of the workers
    """
    if executor is None:
        if num_workers > 1:
            raise ValueError("An executor must be provided to run multiple jobs in parallel")
        executor = SequentialExecutor()
    jobs: tp.List[tp.JobLike[utils.Selector]] = []
    bench = BenchmarkChunk(name=experiment_name, seed=seed, cap_index=cap_index)
    # instanciate the experiment iterator once (in case data needs to be downloaded (MLDA))
    next(registry[experiment_name]())
    # run
    for chunk in bench.split(num_workers):
        # split experiment this way to avoid one job running most slow settings
        jobs.append(executor.submit(chunk.compute, print_function))
    return jobs


# pylint: disable=too-many-arguments
def compute(
    experiment_name: str,
    num_workers: int = 1,
    seed: tp.Optional[int] = None,
    executor: tp.Optional[tp.ExecutorLike] = None,
    print_function: tp.Optional[tp.Callable[[tp.Dict[str, tp.Any]], None]] = None,
    cap_index: tp.Optional[int] = None,
) -> utils.Selector:
    """Submits a job for computation

    Parameters
    ----------
    experiment_name: str
        name of the experiment plan (must be registered in experiments.registry)
    num_workers: int
        number of workers onto which the jobs will be distributed
    seed: int
        a seed for the experiment plan (if seedable)
    executor: Executor-like object
        an object such as concurrent.futures.ThreadPoolExecutor for running experiments in parallel
    print_function: tp.Callable
        a function to print at the end of each experiment (for custom logging)
    cap_index: int
        index at which the experiment plan must be stopped (convenient for testing if the experiment
        plan holds 10k experiment, we can select the first cap_index=100 for instance)

    Returns
    -------
    pd.DataFrame
        The dataframe summarizing all the experiments (each experiment is a line)
    """
    # pylint: disable=unused-argument
    jobs = _submit_jobs(**locals())
    dfs = [j.result() for j in jobs]
    return utils.Selector(pd.concat(dfs))
