# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import tempfile
import warnings
from pathlib import Path
from functools import partial
from unittest import SkipTest
from unittest.mock import patch
from typing import Type, Union, Generator, List
import pytest
import numpy as np
import pandas as pd
from bayes_opt.util import acq_max
import nevergrad as ng
from .. import instrumentation as inst
from ..common.typetools import ArrayLike
from ..common import testing
from . import base
from . import optimizerlib as optlib
from .recaster import FinishedUnderlyingOptimizerWarning
from .optimizerlib import registry


class Fitness:
    """Simple quadratic fitness function which can be used with dimension up to 4
    """

    def __init__(self, x0: ArrayLike) -> None:
        self.x0 = np.array(x0, copy=True)

    def __call__(self, x: ArrayLike) -> float:
        assert len(self.x0) == len(x)
        return float(np.sum((np.array(x, copy=False) - self.x0) ** 2))


def check_optimizer(optimizer_cls: Union[base.OptimizerFamily, Type[base.Optimizer]], budget: int = 300, verify_value: bool = True) -> None:
    # recast optimizer do not support num_workers > 1, and respect no_parallelization.
    num_workers = 1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2
    num_attempts = 1 if not verify_value else 2  # allow 2 attemps to get to the optimum (shit happens...)
    optimum = [0.5, -0.8]
    if optimizer_cls in (optlib.PBIL,):
        optimum = [0, 1, 0, 1, 0, 1]
    fitness = Fitness(optimum)
    for k in range(1, num_attempts + 1):
        optimizer = optimizer_cls(instrumentation=len(optimum), budget=budget, num_workers=num_workers)
        with warnings.catch_warnings():
            # tests do not need to be efficient
            warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
            # some optimizers finish early
            warnings.filterwarnings("ignore", category=FinishedUnderlyingOptimizerWarning)
            # now optimize :)
            candidate = optimizer.minimize(fitness)
        if verify_value and "chain" not in str(optimizer_cls):
            try:
                np.testing.assert_array_almost_equal(candidate.data, optimum, decimal=1)
            except AssertionError as e:
                print(f"Attemp #{k}: failed with best point {tuple(candidate.data)}")
                if k == num_attempts:
                    raise e
            else:
                break
    # check population queue
    if hasattr(optimizer, "population"):  # TODO add a PopBasedOptimizer
        assert len(optimizer.population._queue) == len(set(optimizer.population._queue)), "Queue has duplicated items"  # type: ignore
    # make sure we are correctly tracking the best values
    archive = optimizer.archive
    assert optimizer.current_bests["pessimistic"].pessimistic_confidence_bound == min(
        v.pessimistic_confidence_bound for v in archive.values()
    )
    # add a random point to test tell_not_asked
    assert not optimizer._asked, "All `ask`s  should have been followed by a `tell`"
    try:
        candidate = optimizer.create_candidate.from_data(np.random.normal(0, 1, size=optimizer.dimension))
        optimizer.tell(candidate, 12.0)
    except Exception as e:  # pylint: disable=broad-except
        if not isinstance(e, base.TellNotAskedNotSupportedError):
            raise AssertionError(
                "Optimizers should raise base.TellNotAskedNotSupportedError " "at when telling unasked points if they do not support it"
            ) from e
    else:
        assert optimizer.num_tell == budget + 1
        assert optimizer.num_tell_not_asked == 1


SLOW = [
    "NoisyDE",
    "NoisyBandit",
    "SPSA",
    "NoisyOnePlusOne",
    "OptimisticNoisyOnePlusOne",
    "ASCMADEthird",
    "ASCMA2PDEthird",
    "MultiScaleCMA",
    "PCEDA",
    "MPCEDA",
    "EDA",
    "MEDA",
    "MicroCMA",
]
DISCRETE = ["PBIL", "cGA"]
UNSEEDABLE: List[str] = []


@pytest.mark.parametrize("name", [name for name in registry])  # type: ignore
def test_optimizers(name: str) -> None:
    optimizer_cls = registry[name]
    if isinstance(optimizer_cls, base.OptimizerFamily):
        assert hasattr(optlib, name)  # make sure registration matches name in optlib
    verify = not optimizer_cls.one_shot and name not in SLOW and not any(x in name for x in ["BO", "Discrete"])
    # the following context manager speeds up BO tests
    patched = partial(acq_max, n_warmup=10000, n_iter=2)
    with patch("bayes_opt.bayesian_optimization.acq_max", patched):
        check_optimizer(optimizer_cls, budget=300 if "BO" not in name else 2, verify_value=verify)


class RecommendationKeeper:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.recommendations = pd.DataFrame(columns=[f"v{k}" for k in range(16)])  # up to 64 values
        if filepath.exists():
            self.recommendations = pd.read_csv(filepath, index_col=0)

    def save(self) -> None:
        # sort and remove unused names
        # then update recommendation file
        names = sorted(x for x in self.recommendations.index if x in registry)
        recom = self.recommendations.loc[names, :]
        recom.iloc[:, :] = np.round(recom, 10)
        recom.to_csv(self.filepath)


@pytest.fixture(scope="module")  # type: ignore
def recomkeeper() -> Generator[RecommendationKeeper, None, None]:
    keeper = RecommendationKeeper(filepath=Path(__file__).parent / "recorded_recommendations.csv")
    yield keeper
    keeper.save()


@pytest.mark.parametrize("name", [name for name in registry])  # type: ignore
def test_optimizers_suggest(name: str) -> None:  # pylint: disable=redefined-outer-name
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.simplefilter("ignore", category=base.InefficientSettingsWarning)
        optimizer = registry[name](instrumentation=4, budget=2)
        optimizer.suggest(np.array([12.0] * 4))
        candidate = optimizer.ask()
        try:
            optimizer.tell(candidate, 12)
        except base.TellNotAskedNotSupportedError:
            pass


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("with_parameter", [True, False])  # type: ignore
@pytest.mark.parametrize("name", [name for name in registry])  # type: ignore
def test_optimizers_recommendation(with_parameter: bool,
                                   name: str,
                                   recomkeeper: RecommendationKeeper) -> None:
    # set up environment
    optimizer_cls = registry[name]
    if name in UNSEEDABLE:
        raise SkipTest("Not playing nicely with the tests (unseedable)")
    np.random.seed(None)
    if optimizer_cls.recast or "SplitOptimizer" in name:
        np.random.seed(12)
        random.seed(12)  # may depend on non numpy generator
    # budget=6 by default, larger for special cases needing more
    budget = {"WidePSO": 100, "PSO": 100, "MEDA": 100, "EDA": 100, "MPCEDA": 100, "TBPSA": 100}.get(name, 6)
    if isinstance(optimizer_cls, optlib.DifferentialEvolution):
        budget = 80
    dimension = min(16, max(4, int(np.sqrt(budget))))
    # set up problem
    fitness = Fitness([0.5, -0.8, 0, 4] + (5 * np.cos(np.arange(dimension - 4))).tolist())
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        param: Union[int, ng.p.Instrumentation] = (dimension if not with_parameter else
                                                   ng.p.Instrumentation(ng.p.Array(shape=(dimension,))))
        optim = optimizer_cls(instrumentation=param, budget=budget, num_workers=1)
        optim.instrumentation.random_state.seed(12)
        np.testing.assert_equal(optim.name, name)
        # the following context manager speeds up BO tests
        # BEWARE: BO tests are deterministic but can get different results from a computer to another.
        # Reducing the precision could help in this regard.
        patched = partial(acq_max, n_warmup=10000, n_iter=2)
        with patch("bayes_opt.bayesian_optimization.acq_max", patched):
            candidate = optim.minimize(fitness)
    if name not in recomkeeper.recommendations.index:
        recomkeeper.recommendations.loc[name, :dimension] = tuple(candidate.data)
        raise ValueError(f'Recorded the value for optimizer "{name}", please rerun this test locally.')
    # BO slightly differs from a computer to another
    decimal = 2 if isinstance(optimizer_cls, optlib.ParametrizedBO) else 5
    np.testing.assert_array_almost_equal(
        candidate.data,
        recomkeeper.recommendations.loc[name, :][:dimension],
        decimal=decimal,
        err_msg="Something has changed, if this is normal, delete the following "
        f"file and rerun to update the values:\n{recomkeeper.filepath}",
    )


@testing.parametrized(
    de=("DE", 10, 10, 30),
    de_w=("DE", 50, 40, 40),
    de1=("OnePointDE", 10, 10, 30),
    de1_w=("OnePointDE", 50, 40, 40),
    dim_d=("AlmostRotationInvariantDEAndBigPop", 50, 40, 51),
    dim=("AlmostRotationInvariantDEAndBigPop", 10, 40, 40),
    dim_d_rot=("RotationInvariantDE", 50, 40, 51),
    large=("BPRotationInvariantDE", 10, 40, 70),
)
def test_differential_evolution_popsize(name: str, dimension: int, num_workers: int, expected: int) -> None:
    optim = registry[name](instrumentation=dimension, budget=100, num_workers=num_workers)
    np.testing.assert_equal(optim.llambda, expected)  # type: ignore


def test_portfolio_budget() -> None:
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        for k in range(3, 13):
            optimizer = optlib.Portfolio(instrumentation=2, budget=k)
            np.testing.assert_equal(optimizer.budget, sum(o.budget for o in optimizer.optims))


def test_optimizer_families_repr() -> None:
    Cls = optlib.DifferentialEvolution
    np.testing.assert_equal(repr(Cls()), "DifferentialEvolution()")
    np.testing.assert_equal(repr(Cls(initialization="LHS")), "DifferentialEvolution(initialization='LHS')")
    #
    optimrs = optlib.RandomSearchMaker(cauchy=True)
    np.testing.assert_equal(repr(optimrs), "RandomSearchMaker(cauchy=True)")
    #
    optimso = optlib.ScipyOptimizer(method="COBYLA")
    np.testing.assert_equal(repr(optimso), "ScipyOptimizer(method='COBYLA')")
    assert optimso.no_parallelization


@pytest.mark.parametrize("name", ["PSO", "DE"])  # type: ignore
def test_tell_not_asked(name: str) -> None:
    best = [0.5, -0.8, 0, 4]
    dim = len(best)
    fitness = Fitness(best)
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        opt = optlib.registry[name](instrumentation=dim, budget=2, num_workers=2)
    if name == "PSO":
        opt.llambda = 2  # type: ignore
    else:
        opt._llambda = 2  # type: ignore
    zeros = [0.0] * dim
    opt.tell(opt.create_candidate.from_data(zeros), fitness(zeros))  # not asked
    asked = [opt.ask(), opt.ask()]
    opt.tell(opt.create_candidate.from_data(best), fitness(best))  # not asked
    opt.tell(asked[0], fitness(*asked[0].args))
    opt.tell(asked[1], fitness(*asked[1].args))
    assert opt.num_tell == 4, opt.num_tell
    assert opt.num_ask == 2
    if (0, 0, 0, 0) not in [tuple(x.data) for x in asked]:
        for value in opt.archive.values():
            assert value.count == 1


def test_tbpsa_recom_with_update() -> None:
    budget = 20
    # set up problem
    fitness = Fitness([0.5, -0.8, 0, 4])
    optim = optlib.TBPSA(instrumentation=4, budget=budget, num_workers=1)
    optim.instrumentation.random_state.seed(12)
    optim.llambda = 3
    candidate = optim.minimize(fitness)
    np.testing.assert_almost_equal(candidate.data, [0.037964, 0.0433031, -0.4688667, 0.3633273])


def _square(x: np.ndarray, y: float = 12) -> float:
    return sum((x - 0.5) ** 2) + abs(y)


def test_optimization_doc_instrumentation_example() -> None:
    instrum = inst.Instrumentation(inst.var.Array(2), y=inst.var.Array(1).asscalar())
    optimizer = optlib.OnePlusOne(instrumentation=instrum, budget=100)
    recom = optimizer.minimize(_square)
    assert len(recom.args) == 1
    testing.assert_set_equal(recom.kwargs, ["y"])
    value = _square(*recom.args, **recom.kwargs)
    assert value < 0.2  # should be large enough by an order of magnitude


def test_optimization_discrete_with_one_sample() -> None:
    optimizer = optlib.PortfolioDiscreteOnePlusOne(instrumentation=1, budget=10)
    optimizer.minimize(_square)


@pytest.mark.parametrize("name", ["TBPSA", "PSO", "TwoPointsDE"])  # type: ignore
# this test is added because some generic class (like Population) can fail to be pickled
def test_population_pickle(name: str) -> None:
    # example of work around:
    # "self.population = base.utils.Population[DEParticle]([])"
    # becomes:
    # "self.population: base.utils.Population[DEParticle] = base.utils.Population([])""
    optim = registry[name](instrumentation=12, budget=100, num_workers=2)
    with tempfile.TemporaryDirectory() as folder:
        filepath = Path(folder) / "dump_test.pkl"
        optim.dump(filepath)


def test_bo_instrumentation_and_parameters() -> None:
    # instrumentation
    instrumentation = inst.Instrumentation(ng.p.Choice([True, False]))
    with pytest.warns(base.InefficientSettingsWarning):
        optlib.QRBO(instrumentation, budget=10)
    with pytest.warns(None) as record:
        opt = optlib.ParametrizedBO(gp_parameters={"alpha": 1})(instrumentation, budget=10)
    assert not record, record.list  # no warning
    # parameters
    # make sure underlying BO optimizer gets instantiated correctly
    opt.tell(opt.create_candidate.from_call(True), 0.0)


def test_chaining() -> None:
    budgets = [7, 19]
    optimizer = optlib.Chaining([optlib.LHSSearch, optlib.HaltonSearch, optlib.OnePlusOne], budgets)(2, 40)
    optimizer.minimize(_square)
    expected = [(7, 7, 0), (19, 19 + 7, 7), (14, 14 + 19 + 7, 19 + 7)]
    for (ex_ask, ex_tell, ex_tell_not_asked), opt in zip(expected, optimizer._optimizers):  # type: ignore
        assert opt.num_ask == ex_ask
        assert opt.num_tell == ex_tell
        assert opt.num_tell_not_asked == ex_tell_not_asked
    optimizer.ask()
    assert optimizer._optimizers[-1].num_ask == 15  # type: ignore


def test_instrumentation_optimizer_reproducibility() -> None:
    instrumentation = inst.Instrumentation(ng.p.Array(shape=(1,)), y=ng.p.Choice(list(range(100))))
    instrumentation.random_state.seed(12)
    optimizer = optlib.RandomSearch(instrumentation, budget=10)
    recom = optimizer.minimize(_square)
    np.testing.assert_equal(recom.kwargs["y"], 67)


def test_constrained_optimization() -> None:
    instrumentation = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(instrumentation, budget=100)
    optimizer.instrumentation.random_state.seed(12)
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.instrumentation.register_cheap_constraint(lambda i: i[1]["x"][0] >= 1)  # type:ignore
    recom = optimizer.minimize(_square)
    np.testing.assert_array_almost_equal([recom.kwargs["x"][0], recom.kwargs["y"]], [1.005573e+00, 3.965783e-04])


@pytest.mark.parametrize("name", [name for name in registry])  # type: ignore
def test_parametrization_offset(name: str) -> None:
    if "PSO" in name or "BO" in name:
        raise SkipTest("PSO and BO have large initial variance")
    parametrization = ng.p.Instrumentation(ng.p.Array(init=[1e12, 1e12]))
    optimizer = registry[name](parametrization, budget=100, num_workers=1)
    for k in range(10 if "BO" not in name else 2):
        candidate = optimizer.ask()
        assert candidate.args[0][0] > 100, f"Candidate value[0] at iteration #{k} is below 100: {candidate.args[0][0]}"
        optimizer.tell(candidate, 0)
