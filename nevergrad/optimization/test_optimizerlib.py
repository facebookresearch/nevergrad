# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import time
import random
import logging
import platform
import tempfile
import warnings
from pathlib import Path
from functools import partial
from unittest import SkipTest
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from bayes_opt.util import acq_max
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import base
from . import optimizerlib as optlib
from . import experimentalvariants as xpvariants
from .recaster import FinishedUnderlyingOptimizerWarning
from .optimizerlib import registry


class Fitness:
    """Simple quadratic fitness function which can be used with dimension up to 4
    """

    def __init__(self, x0: tp.ArrayLike) -> None:
        self.x0 = np.array(x0, copy=True)
        self.call_times: tp.List[float] = []

    def __call__(self, x: tp.ArrayLike) -> float:
        assert len(self.x0) == len(x)
        self.call_times.append(time.time())
        return float(np.sum((np.array(x, copy=False) - self.x0) ** 2))

    def get_factors(self) -> tp.Tuple[float, float]:
        logdiffs = np.log(np.maximum(1e-15, np.cumsum(np.diff(self.call_times))))
        nums = np.arange(len(logdiffs))
        slope, intercept = (float(np.exp(x)) for x in stats.linregress(nums, logdiffs)[:2])
        return slope, intercept


def check_optimizer(
        optimizer_cls: tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]],
        budget: int = 300,
        verify_value: bool = True
) -> None:
    # recast optimizer do not support num_workers > 1, and respect no_parallelization.
    num_workers = 1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2
    num_attempts = 1 if not verify_value else 3  # allow 3 attemps to get to the optimum (shit happens...)
    optimum = [0.5, -0.8]
    fitness = Fitness(optimum)
    for k in range(1, num_attempts + 1):
        fitness = Fitness(optimum)
        optimizer = optimizer_cls(parametrization=len(optimum), budget=budget, num_workers=num_workers)
        assert isinstance(optimizer.provide_recommendation(), ng.p.Parameter), "Recommendation should be available from start"
        with warnings.catch_warnings():
            # tests do not need to be efficient
            warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
            # some optimizers finish early
            warnings.filterwarnings("ignore", category=FinishedUnderlyingOptimizerWarning)
            # skip BO error on windows (issue #506)
            if "BO" in optimizer.name:
                raise SkipTest("BO is currently not well supported")
            if "Many" in optimizer.name:
                raise SkipTest("When many algorithms are in the portfolio we are not good for small budget.")
            # now optimize :)
            candidate = optimizer.minimize(fitness)
        if verify_value and "chain" not in str(optimizer_cls):
            try:
                np.testing.assert_array_almost_equal(candidate.args[0], optimum, decimal=1)
            except AssertionError as e:
                print(f"Attemp #{k}: failed with best point {tuple(candidate.args[0])}")
                if k == num_attempts:
                    raise e
            else:
                break
    if budget > 100:
        slope, intercept = fitness.get_factors()
        print(f"For your information: slope={slope} and intercept={intercept}")
    # make sure we are correctly tracking the best values
    archive = optimizer.archive
    assert optimizer.current_bests["pessimistic"].pessimistic_confidence_bound == min(
        v.pessimistic_confidence_bound for v in archive.values()
    )
    # add a random point to test tell_not_asked
    assert not optimizer._asked, "All `ask`s  should have been followed by a `tell`"
    try:
        data = np.random.normal(0, 1, size=optimizer.dimension)
        candidate = optimizer.parametrization.spawn_child().set_standardized_data(data, deterministic=False)
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
    "EDA",
    "MicroCMA",
    "ES",
]
UNSEEDABLE: tp.List[str] = []


@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers(name: str) -> None:
    optimizer_cls = registry[name]
    if isinstance(optimizer_cls, base.ConfiguredOptimizer):
        assert any(hasattr(mod, name) for mod in (optlib, xpvariants))  # make sure registration matches name in optlib/xpvariants
        assert optimizer_cls.__class__(**optimizer_cls._config) == optimizer_cls, "Similar configuration are not equal"
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
def recomkeeper() -> tp.Generator[RecommendationKeeper, None, None]:
    keeper = RecommendationKeeper(filepath=Path(__file__).parent / "recorded_recommendations.csv")
    yield keeper
    keeper.save()


@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_suggest(name: str) -> None:  # pylint: disable=redefined-outer-name
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.simplefilter("ignore", category=base.InefficientSettingsWarning)
        optimizer = registry[name](parametrization=4, budget=2)
        optimizer.suggest(np.array([12.0] * 4))
        candidate = optimizer.ask()
        try:
            optimizer.tell(candidate, 12)
            # The optimizer should recommend its suggestion, except for a few optimization methods:
            if name not in ["SPSA", "TBPSA", "StupidRandom"]:
                np.testing.assert_array_almost_equal(optimizer.provide_recommendation().value, [12.0] * 4)
        except base.TellNotAskedNotSupportedError:
            pass


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_recommendation(name: str, recomkeeper: RecommendationKeeper) -> None:
    if "BO" in name:
        raise SkipTest("BO not cool these days for some reason!")
    # set up environment
    optimizer_cls = registry[name]
    if name in UNSEEDABLE:
        raise SkipTest("Not playing nicely with the tests (unseedable)")
    np.random.seed(None)
    if optimizer_cls.recast or "SplitOptimizer" in name:
        np.random.seed(12)
        random.seed(12)  # may depend on non numpy generator
    # budget=6 by default, larger for special cases needing more
    budget = {"WidePSO": 100, "PSO": 200, "MEDA": 100, "EDA": 100, "MPCEDA": 100, "TBPSA": 100}.get(name, 6)
    if isinstance(optimizer_cls, (optlib.DifferentialEvolution, optlib.EvolutionStrategy)):
        budget = 80
    dimension = min(16, max(4, int(np.sqrt(budget))))
    # set up problem
    fitness = Fitness([0.5, -0.8, 0, 4] + (5 * np.cos(np.arange(dimension - 4))).tolist())
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        optim = optimizer_cls(parametrization=dimension, budget=budget, num_workers=1)
        optim.parametrization.random_state.seed(12)
        np.testing.assert_equal(optim.name, name)
        # the following context manager speeds up BO tests
        # BEWARE: BO tests are deterministic but can get different results from a computer to another.
        # Reducing the precision could help in this regard.
        patched = partial(acq_max, n_warmup=10000, n_iter=2)
        with patch("bayes_opt.bayesian_optimization.acq_max", patched):
            recom = optim.minimize(fitness)
    if name not in recomkeeper.recommendations.index:
        recomkeeper.recommendations.loc[name, :dimension] = tuple(recom.value)
        raise ValueError(f'Recorded the value for optimizer "{name}", please rerun this test locally.')
    # BO slightly differs from a computer to another
    decimal = 2 if isinstance(optimizer_cls, optlib.ParametrizedBO) or "BO" in name else 5
    np.testing.assert_array_almost_equal(
        recom.value,
        recomkeeper.recommendations.loc[name, :][:dimension],
        decimal=decimal,
        err_msg="Something has changed, if this is normal, delete the following "
        f"file and rerun to update the values:\n{recomkeeper.filepath}",
    )
    # check that by default the recommendation has been evaluated
    if isinstance(optimizer_cls, optlib.EvolutionStrategy):  # no noisy variants
        assert recom.loss is not None


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
    optim = registry[name](parametrization=dimension, budget=100, num_workers=num_workers)
    np.testing.assert_equal(optim.llambda, expected)  # type: ignore


def test_portfolio_budget() -> None:
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        for k in range(3, 13):
            optimizer = optlib.Portfolio(parametrization=2, budget=k)
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
    #
    optimcma = optlib.ParametrizedCMA(diagonal=True)
    np.testing.assert_equal(repr(optimcma), "ParametrizedCMA(diagonal=True)")


@pytest.mark.parametrize("name", ["PSO", "DE"])  # type: ignore
def test_tell_not_asked(name: str) -> None:
    param = ng.p.Scalar()
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        opt = optlib.registry[name](parametrization=param, budget=2, num_workers=2)
    opt.llambda = 2  # type: ignore
    t_10 = opt.parametrization.spawn_child(new_value=10)
    t_100 = opt.parametrization.spawn_child(new_value=100)
    assert not opt.population  # type: ignore
    opt.tell(t_10, 90)  # not asked
    assert len(opt.population) == 1  # type: ignore
    asked = opt.ask()
    opt.tell(asked, 88)
    assert len(opt.population) == 2  # type: ignore
    opt.tell(t_100, 0)  # not asked
    asked = opt.ask()
    opt.tell(asked, 89)
    assert len(opt.population) == 2  # type: ignore
    assert opt.num_tell == 4, opt.num_tell
    assert opt.num_ask == 2
    assert len(opt.population) == 2  # type: ignore
    assert int(opt.recommend().value) == 100
    if isinstance(opt.population, dict):  # type: ignore
        assert t_100.uid in opt.population  # type: ignore
    for point, value in opt.archive.items_as_arrays():
        assert value.count == 1, f"Error for point {point}"


def test_tbpsa_recom_with_update() -> None:
    budget = 20
    # set up problem
    fitness = Fitness([0.5, -0.8, 0, 4])
    optim = optlib.TBPSA(parametrization=4, budget=budget, num_workers=1)
    optim.parametrization.random_state.seed(12)
    optim.popsize.llambda = 3  # type: ignore
    candidate = optim.minimize(fitness)
    np.testing.assert_almost_equal(candidate.args[0], [0.037964, 0.0433031, -0.4688667, 0.3633273])


def _square(x: np.ndarray, y: float = 12) -> float:
    return sum((x - 0.5) ** 2) + abs(y)


def test_optimization_doc_parametrization_example() -> None:
    instrum = ng.p.Instrumentation(ng.p.Array(shape=(2,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(parametrization=instrum, budget=100)
    recom = optimizer.minimize(_square)
    assert len(recom.args) == 1
    testing.assert_set_equal(recom.kwargs, ["y"])
    value = _square(*recom.args, **recom.kwargs)
    assert value < 0.2  # should be large enough by an order of magnitude


def test_optimization_discrete_with_one_sample() -> None:
    optimizer = xpvariants.PortfolioDiscreteOnePlusOne(parametrization=1, budget=10)
    optimizer.minimize(_square)


@pytest.mark.parametrize("name", ["TBPSA", "PSO", "TwoPointsDE"])  # type: ignore
# this test is added because some generic class can fail to be pickled
def test_population_pickle(name: str) -> None:
    # example of work around:
    # "self.population = base.utils.Population[DEParticle]([])"
    # becomes:
    # "self.population: base.utils.Population[DEParticle] = base.utils.Population([])""
    optim = registry[name](parametrization=12, budget=100, num_workers=2)
    with tempfile.TemporaryDirectory() as folder:
        filepath = Path(folder) / "dump_test.pkl"
        optim.dump(filepath)


def test_bo_parametrization_and_parameters() -> None:
    # parametrization
    parametrization = ng.p.Instrumentation(ng.p.Choice([True, False]))
    with pytest.warns(base.InefficientSettingsWarning):
        xpvariants.QRBO(parametrization, budget=10)
    with pytest.warns(None) as record:
        opt = optlib.ParametrizedBO(gp_parameters={"alpha": 1})(parametrization, budget=10)
    assert not record, record.list  # no warning
    # parameters
    # make sure underlying BO optimizer gets instantiated correctly
    new_candidate = opt.parametrization.spawn_child(new_value=((True,), {}))
    opt.tell(new_candidate, 0.0)


def test_chaining() -> None:
    budgets = [7, 19]
    optimizer = optlib.Chaining([optlib.LHSSearch, optlib.HaltonSearch, optlib.OnePlusOne], budgets)(2, 40)
    optimizer.minimize(_square)
    expected = [(7, 7, 0), (19, 19 + 7, 7), (14, 14 + 19 + 7, 19 + 7)]
    for (ex_ask, ex_tell, ex_tell_not_asked), opt in zip(expected, optimizer.optimizers):  # type: ignore
        assert opt.num_ask == ex_ask
        assert opt.num_tell == ex_tell
        assert opt.num_tell_not_asked == ex_tell_not_asked
    optimizer.ask()
    assert optimizer.optimizers[-1].num_ask == 15  # type: ignore


def test_parametrization_optimizer_reproducibility() -> None:
    parametrization = ng.p.Instrumentation(ng.p.Array(shape=(1,)), y=ng.p.Choice(list(range(100))))
    parametrization.random_state.seed(12)
    optimizer = optlib.RandomSearch(parametrization, budget=10)
    recom = optimizer.minimize(_square)
    np.testing.assert_equal(recom.kwargs["y"], 4)
    # resampling deterministically
    # (this test has been reeeally useful so far, any change of the output must be investigated)
    data = recom.get_standardized_data(reference=optimizer.parametrization)
    recom = optimizer.parametrization.spawn_child().set_standardized_data(data, deterministic=True)
    np.testing.assert_equal(recom.kwargs["y"], 67)


def test_parallel_es() -> None:
    opt = optlib.EvolutionStrategy(popsize=3, offsprings=None)(4, budget=20, num_workers=5)
    for k in range(35):
        cand = opt.ask()  # asking should adapt to the parallelization
        if not k:
            opt.tell(cand, 1)


def test_constrained_optimization() -> None:
    parametrization = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(parametrization, budget=100)
    optimizer.parametrization.random_state.seed(12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.parametrization.register_cheap_constraint(lambda i: i[1]["x"][0] >= 1)  # type:ignore
    recom = optimizer.minimize(_square)
    np.testing.assert_array_almost_equal([recom.kwargs["x"][0], recom.kwargs["y"]], [1.005573e+00, 3.965783e-04])


@pytest.mark.parametrize("name", registry)  # type: ignore
def test_parametrization_offset(name: str) -> None:
    if "PSO" in name or "BO" in name:
        raise SkipTest("PSO and BO have large initial variance")
    if "Cobyla" in name and platform.system() == "Windows":
        raise SkipTest("Cobyla is flaky on Windows for unknown reasons")
    parametrization = ng.p.Instrumentation(ng.p.Array(init=[1e12, 1e12]))
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        optimizer = registry[name](parametrization, budget=100, num_workers=1)
    for k in range(10 if "BO" not in name else 2):
        candidate = optimizer.ask()
        assert candidate.args[0][0] > 100, f"Candidate value[0] at iteration #{k} is below 100: {candidate.value}"
        optimizer.tell(candidate, 0)


def test_optimizer_sequence() -> None:
    budget = 24
    parametrization = ng.p.Tuple(*(ng.p.Scalar(lower=-12, upper=12) for _ in range(2)))
    optimizer = optlib.LHSSearch(parametrization, budget=24)
    points = [np.array(optimizer.ask().value) for _ in range(budget)]
    assert sum(any(abs(x) > 11 for x in p) for p in points) > 0


def test_shiwa_dim1() -> None:
    param = ng.p.Log(lower=1, upper=1000).set_integer_casting()
    init = param.value
    optimizer = optlib.Shiwa(param, budget=40)
    recom = optimizer.minimize(np.abs)
    assert recom.value < init


@pytest.mark.parametrize(  # type: ignore
    "name,param,budget,num_workers,expected",
    [("Shiwa", 1, 10, 1, "Cobyla"),
     ("Shiwa", 1, 10, 2, "CMA"),
     ("Shiwa", ng.p.Log(lower=1, upper=1000).set_integer_casting(), 10, 2, "DoubleFastGADiscreteOnePlusOne"),
     ("NGOpt", 1, 10, 1, "MetaModel"),
     ("NGOpt", 1, 10, 2, "MetaModel"),
     ("NGOpt", ng.p.Log(lower=1, upper=1000).set_integer_casting(), 10, 2, "DoubleFastGADiscreteOnePlusOne"),
     ("NGOpt", ng.p.TransitionChoice(range(30), repetitions=10), 10, 2, "DiscreteBSOOnePlusOne"),
     ("NGOpt", ng.p.TransitionChoice(range(3), repetitions=10), 10, 2, "CMandAS2"),
     ("NGO", 1, 10, 1, "Cobyla"),
     ("NGO", 1, 10, 2, "CMA"),
     ]  # pylint: disable=too-many-arguments
)
def test_shiwa_selection(name: str, param: tp.Any, budget: int, num_workers: int, expected: str, caplog: tp.Any) -> None:
    with caplog.at_level(logging.DEBUG, logger="nevergrad.optimization.optimizerlib"):
        optlib.registry[name](param, budget=budget, num_workers=num_workers)
        pattern = rf".*{name} selected (?P<name>\w+?) optimizer\."
        match = re.match(pattern, caplog.text, re.MULTILINE)
        assert match is not None, f"Did not detect selection in logs: {caplog.text}"
        assert match.group("name") == expected


def test_bo_ordering() -> None:
    optim = ng.optimizers.ParametrizedBO(initialization='Hammersley')(
        parametrization=ng.p.Choice(range(12)),
        budget=10
    )
    cand = optim.ask()
    optim.tell(cand, 12)
