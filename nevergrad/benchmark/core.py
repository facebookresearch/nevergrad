# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Iterator, Tuple, Iterable
import numpy as np
import pandas as pd
from .experiments import registry, Experiment
from ..common import tools
from ..common.typetools import ExecutorLike, JobLike, PathLike
from ..optimization.utils import SequentialExecutor


def import_additional_module(filepath: PathLike) -> None:
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
    spec.loader.exec_module(module)


def save_or_append_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Saves a dataframe to a file in append mode
    """
    if path.exists():
        print("Appending to existing file")
        predf = pd.read_csv(str(path))
        df = pd.concat([predf, df], sort=False)
    df.to_csv(path, index=False)


class Moduler:

    def __init__(self, modulo: int, index: int) -> None:
        assert modulo > 0, "Modulo must be strictly positive"
        assert index < modulo, "Index must be strictly smaller than modulo"
        self.modulo = modulo
        self.index = index

    def __call__(self, index: int) -> bool:
        return (index % self.modulo) == self.index

    def __repr__(self) -> str:
        return f"Moduler({self.index}, {self.modulo}"


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

    def __init__(self, name: str, repetitions: int = 1, seed: Optional[int] = None, cap_index: Optional[int] = None) -> None:
        self.name = name
        self.seed = seed
        self.cap_index = None if cap_index is None else max(1, int(cap_index))
        self.moduler = Moduler(1, 0)
        self.repetitions = repetitions
        self.summaries: List[Dict[str, Any]] = []
        self._id = (datetime.datetime.now().strftime("%y-%m-%d_%H%M") + "_" +
                    "".join(np.random.choice([x for x in "abcdefghijklmnopqrstuvwxyz"], 4)))

    @property
    def id(self) -> str:
        """Unique ID which can be used to print in a file for instance
        """
        return f"{self._id}_i{self.moduler.index}m{self.moduler.modulo}"

    def __iter__(self) -> Iterator[Tuple[int, Experiment]]:
        maker = registry[self.name]
        seeds: Iterable[Optional[int]] = ((None for _ in range(self.repetitions)) if self.seed is None else
                                          range(self.seed, self.seed + self.repetitions))
        # check experiments.py to see seedable xp
        generators = [maker() if seed is None else maker(seed=seed) for seed in seeds]
        if self.cap_index is not None:
            generators = [itertools.islice(g, 0, self.cap_index) for g in generators]
        enumerated_selection = ((k, s) for (k, s) in enumerate(itertools.chain.from_iterable(generators)) if self.moduler(k))
        return enumerated_selection

    def split(self, number: int) -> List['BenchmarkChunk']:
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
        for k in range(number):
            chunk = BenchmarkChunk(name=self.name, repetitions=self.repetitions, seed=self.seed, cap_index=self.cap_index)
            chunk.moduler = Moduler(self.moduler.modulo * number, self.moduler.index + k * self.moduler.modulo)
            chunk._id = self._id
            chunks.append(chunk)
        return chunks

    def __repr__(self) -> str:
        return f"BenchmarkChunk({self.name}, {self.repetitions}, {self.seed}) with {self.moduler}"

    def compute(self, process_function: Optional[Callable[["BenchmarkChunk", Experiment], None]] = None) -> tools.Selector:
        """Run all the experiments and returns the result dataframe.

        Parameters
        ----------
        print_function: Callable
            a function to print at the end of each experiment (for custom logging)
        """
        for local_ind, (index, xp) in enumerate(self):
            if local_ind < len(self.summaries):
                continue  # already computed
            print(f"Starting {index}: {xp}", flush=True)
            xp.run()
            summary = xp.get_description()
            if process_function is not None:
                process_function(self, xp)
            self.summaries.append(summary)
            print(f"Finished {index}", flush=True)
        return tools.Selector(data=self.summaries)


# pylint: disable=too-many-arguments
def _submit_jobs(experiment_name: str, num_workers: int = 1, seed: Optional[int] = None, executor: Optional[ExecutorLike] = None,
                 print_function: Optional[Callable[[Experiment], None]] = None, cap_index: Optional[int] = None) -> List[JobLike]:
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
    print_function: Callable
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
    jobs: List[JobLike] = []
    bench = BenchmarkChunk(name=experiment_name, seed=seed, cap_index=cap_index)
    for chunk in bench.split(num_workers):
        # split experiment this way to avoid one job running most slow settings
        jobs.append(executor.submit(chunk.compute, print_function))
    return jobs


# pylint: disable=too-many-arguments
def compute(experiment_name: str, num_workers: int = 1, seed: Optional[int] = None, executor: Optional[ExecutorLike] = None,
            print_function: Optional[Callable[[Dict[str, Any]], None]] = None, cap_index: Optional[int] = None) -> tools.Selector:
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
    print_function: Callable
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
    return tools.Selector(pd.concat(dfs))
