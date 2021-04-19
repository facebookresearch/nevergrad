# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
import time
import random
import inspect
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
from nevergrad.common import errors
from . import base
from . import optimizerlib as optlib
from . import experimentalvariants as xpvariants
from . import es
from .optimizerlib import registry
from .optimizerlib import NGOptBase


# decorators to be used when testing on Windows is unecessary
# or cumbersome
skip_win_perf = pytest.mark.skipif(
    sys.platform == "win32", reason="Slow, and no need to test performance on all platforms"
)


class Fitness:
    """Simple quadratic fitness function which can be used with dimension up to 4"""

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


# pylint: disable=too-many-locals
def check_optimizer(
    optimizer_cls: tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]],
    budget: int = 300,
    verify_value: bool = True,
) -> None:
    # recast optimizer do not support num_workers > 1, and respect no_parallelization.
    num_workers = 1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2
    num_attempts = 1 if not verify_value else 3  # allow 3 attemps to get to the optimum (shit happens...)
    optimum = [0.5, -0.8]
    fitness = Fitness(optimum)
    for k in range(1, num_attempts + 1):
        fitness = Fitness(optimum)
        optimizer = optimizer_cls(parametrization=len(optimum), budget=budget, num_workers=num_workers)
        assert isinstance(
            optimizer.provide_recommendation(), ng.p.Parameter
        ), "Recommendation should be available from start"
        with testing.suppress_nevergrad_warnings():
            candidate = optimizer.minimize(fitness)
        raised = False
        if verify_value:
            try:
                np.testing.assert_array_almost_equal(candidate.args[0], optimum, decimal=1)
            except AssertionError as e:
                raised = True
                print(f"Attemp #{k}: failed with best point {tuple(candidate.args[0])}")
                if k == num_attempts:
                    raise e
        if not raised:
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
        candidate = optimizer.parametrization.spawn_child().set_standardized_data(data)
        optimizer.tell(candidate, 12.0)
    except Exception as e:  # pylint: disable=broad-except
        if not isinstance(e, base.errors.TellNotAskedNotSupportedError):
            raise AssertionError(
                "Optimizers should raise base.TellNotAskedNotSupportedError "
                "at when telling unasked points if they do not support it"
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


def buggy_function(x: np.ndarray) -> float:
    if any(x[::2] > 0.0):
        return float("nan")
    if any(x > 0.0):
        return float("inf")
    return np.sum(x ** 2)


@skip_win_perf  # type: ignore
@pytest.mark.parametrize("name", registry)  # type: ignore
@testing.suppress_nevergrad_warnings()  # hides bad loss
def test_infnan(name: str) -> None:
    optim_cls = registry[name]
    optim = optim_cls(parametrization=2, budget=70)
    if not (
        any(
            x in name
            for x in [
                "EDA",
                "EMNA",
                "Stupid",
                "Large",
                "TBPSA",
                "BO",
                "Noisy",
                "Chain",
                "chain",  # TODO: remove when possible
            ]
        )
    ):
        recom = optim.minimize(buggy_function)
        result = buggy_function(recom.value)
        if result < 2.0:
            return
        assert (  # The "bad" algorithms, most of them originating in CMA's recommendation rule.
            any(x == name for x in ["WidePSO", "SPSA", "NGOptBase", "Shiwa", "NGO"])
            or isinstance(optim, (optlib.Portfolio, optlib._CMA, optlib.recaster.SequentialRecastOptimizer))
            or "NGOpt" in name
        )  # Second chance!
        recom = optim.minimize(buggy_function)
        result = buggy_function(recom.value)
        result < 2.0, f"{name} failed and got {result} with {recom.value} (type is {type(optim)})."


@skip_win_perf  # type: ignore
@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers(name: str) -> None:
    """Checks that each optimizer is able to converge on a simple test case"""
    optimizer_cls = registry[name]
    if isinstance(optimizer_cls, base.ConfiguredOptimizer):
        assert any(
            hasattr(mod, name) for mod in (optlib, xpvariants)
        )  # make sure registration matches name in optlib/xpvariants
        assert (
            optimizer_cls.__class__(**optimizer_cls._config) == optimizer_cls
        ), "Similar configuration are not equal"
    # some classes of optimizer are eigher slow or not good with small budgets:
    nameparts = ["Many", "Chain", "BO", "Discrete"] + ["chain"]  # TODO remove chain when possible
    is_ngopt = inspect.isclass(optimizer_cls) and issubclass(optimizer_cls, NGOptBase)  # type: ignore
    verify = (
        not optimizer_cls.one_shot
        and name not in SLOW
        and not any(x in name for x in nameparts)
        and not is_ngopt
    )
    budget = 300 if "BO" not in name and not is_ngopt else 4
    # the following context manager speeds up BO tests
    patched = partial(acq_max, n_warmup=10000, n_iter=2)
    with patch("bayes_opt.bayesian_optimization.acq_max", patched):
        check_optimizer(optimizer_cls, budget=budget, verify_value=verify)


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


@testing.suppress_nevergrad_warnings()
@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_suggest(name: str) -> None:  # pylint: disable=redefined-outer-name
    optimizer = registry[name](parametrization=4, budget=2)
    optimizer.suggest(np.array([12.0] * 4))
    candidate = optimizer.ask()
    try:
        optimizer.tell(candidate, 12)
        # The optimizer should recommend its suggestion, except for a few optimization methods:
        if name not in ["SPSA", "TBPSA", "StupidRandom"]:
            np.testing.assert_array_almost_equal(optimizer.provide_recommendation().value, [12.0] * 4)
    except base.errors.TellNotAskedNotSupportedError:
        pass


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_recommendation(name: str, recomkeeper: RecommendationKeeper) -> None:
    if name in UNSEEDABLE:
        raise SkipTest("Not playing nicely with the tests (unseedable)")
    if "BO" in name:
        raise SkipTest("BO differs from one computer to another")
    # set up environment
    optimizer_cls = registry[name]
    np.random.seed(None)
    if optimizer_cls.recast:
        np.random.seed(12)
        random.seed(12)  # may depend on non numpy generator
    # budget=6 by default, larger for special cases needing more
    budget = {"WidePSO": 100, "PSO": 200, "MEDA": 100, "EDA": 100, "MPCEDA": 100, "TBPSA": 100}.get(name, 6)
    if isinstance(optimizer_cls, (optlib.DifferentialEvolution, optlib.EvolutionStrategy)):
        budget = 80
    dimension = min(16, max(4, int(np.sqrt(budget))))
    # set up problem
    fitness = Fitness([0.5, -0.8, 0, 4] + (5 * np.cos(np.arange(dimension - 4))).tolist())
    with testing.suppress_nevergrad_warnings():
        optim = optimizer_cls(parametrization=dimension, budget=budget, num_workers=1)
        optim.parametrization.random_state.seed(12)
        np.testing.assert_equal(optim.name, name)
        # the following context manager speeds up BO tests
        # BEWARE: BO tests are deterministic but can get different results from a computer to another.
        # Reducing the precision could help in this regard.
        # patched = partial(acq_max, n_warmup=10000, n_iter=2)
        # with patch("bayes_opt.bayesian_optimization.acq_max", patched):
        recom = optim.minimize(fitness)
    if name not in recomkeeper.recommendations.index:
        recomkeeper.recommendations.loc[name, :dimension] = tuple(recom.value)
        raise ValueError(
            f'Recorded the value {tuple(recom.value)} for optimizer "{name}", please rerun this test locally.'
        )
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


@testing.suppress_nevergrad_warnings()
def test_portfolio_budget() -> None:
    for k in range(3, 13):
        optimizer = optlib.Portfolio(parametrization=2, budget=k)
        np.testing.assert_equal(optimizer.budget, sum(o.budget for o in optimizer.optims))


def test_optimizer_families_repr() -> None:
    Cls = optlib.DifferentialEvolution
    np.testing.assert_equal(repr(Cls()), "DifferentialEvolution()")
    np.testing.assert_equal(repr(Cls(initialization="LHS")), "DifferentialEvolution(initialization='LHS')")
    #
    optimrs = optlib.RandomSearchMaker(sampler="cauchy")
    np.testing.assert_equal(repr(optimrs), "RandomSearchMaker(sampler='cauchy')")
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
    with testing.suppress_nevergrad_warnings():
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
    assert value < 0.25  # should be large enough by an order of magnitude (but is not :s)


def test_optimization_discrete_with_one_sample() -> None:
    optimizer = optlib.PortfolioDiscreteOnePlusOne(parametrization=1, budget=10)
    optimizer.minimize(_square)


@pytest.mark.parametrize("name", ["TBPSA", "PSO", "TwoPointsDE", "CMA", "BO"])  # type: ignore
def test_optim_pickle(name: str) -> None:
    # some generic class can fail to be pickled:
    # example of work around:
    # "self.population = base.utils.Population[DEParticle]([])"
    # becomes:
    # "self.population: base.utils.Population[DEParticle] = base.utils.Population([])""
    #
    # Scipy optimizers also fail to be pickled, but this is more complex to solve (not supported yet)
    optim = registry[name](parametrization=12, budget=100, num_workers=2)
    with tempfile.TemporaryDirectory() as folder:
        optim.dump(Path(folder) / "dump_test.pkl")


def test_bo_parametrization_and_parameters() -> None:
    # parametrization
    parametrization = ng.p.Instrumentation(ng.p.Choice([True, False]))
    with pytest.warns(errors.InefficientSettingsWarning):
        xpvariants.QRBO(parametrization, budget=10)
    with pytest.warns(None) as record:
        opt = optlib.ParametrizedBO(gp_parameters={"alpha": 1})(parametrization, budget=10)
    assert not record, record.list  # no warning
    # parameters
    # make sure underlying BO optimizer gets instantiated correctly
    new_candidate = opt.parametrization.spawn_child(new_value=((True,), {}))
    opt.tell(new_candidate, 0.0)


def test_bo_init() -> None:
    arg = ng.p.Scalar(init=4, lower=1, upper=10).set_integer_casting()
    # The test was flaky with normalize_y=True.
    gp_param = {"alpha": 1e-5, "normalize_y": False, "n_restarts_optimizer": 1, "random_state": None}
    my_opt = ng.optimizers.ParametrizedBO(gp_parameters=gp_param, initialization=None)
    optimizer = my_opt(parametrization=arg, budget=10)
    optimizer.minimize(np.abs)


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
    optimizer = optlib.RandomSearch(parametrization, budget=20)
    recom = optimizer.minimize(_square)
    np.testing.assert_equal(recom.kwargs["y"], 1)
    # resampling deterministically, to make sure it is identical
    data = recom.get_standardized_data(reference=optimizer.parametrization)
    recom = optimizer.parametrization.spawn_child()
    with ng.p.helpers.deterministic_sampling(recom):
        recom.set_standardized_data(data)
    np.testing.assert_equal(recom.kwargs["y"], 1)


@testing.suppress_nevergrad_warnings()
def test_parallel_es() -> None:
    opt = optlib.EvolutionStrategy(popsize=3, offsprings=None)(4, budget=20, num_workers=5)
    for k in range(35):
        cand = opt.ask()  # asking should adapt to the parallelization
        if not k:
            opt.tell(cand, 1)


@testing.suppress_nevergrad_warnings()
@skip_win_perf  # type: ignore
@pytest.mark.parametrize(
    "dimension, num_workers, scale, budget, ellipsoid",
    [
        (2, 8, 1.0, 120, False),
        (2, 3, 8.0, 130, True),
        (5, 1, 1.0, 150, False),
        # Interesting tests removed for flakiness:
        # (8, 27, 8., 380, True),
        # (2, 1, 8., 120, True),
        # (2, 3, 8., 70, False),
        # (1, 1, 1., 20, True),
        # (1, 3, 5., 20, False),
        # (2, 3, 1., 70, True),
        # (2, 1, 8., 40, False),
        # (5, 3, 1., 225, True),
        # (5, 1, 8., 150, False),
        # (5, 3, 8., 500, True),
        # (9, 27, 8., 700, True),
        # (10, 27, 8., 400, False),
    ],
)
def test_metamodel(dimension: int, num_workers: int, scale: float, budget: int, ellipsoid: bool) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""

    def _square(x: np.ndarray) -> float:
        return sum((-scale + x) ** 2)

    def _ellips(x: np.ndarray) -> float:
        return sum(((-scale + x) * (np.arange(1, dimension + 1) ** 2)) ** 2)

    _target = _ellips if ellipsoid else _square

    # In both cases we compare MetaModel and CMA for a same given budget.
    # But we expect MetaModel to be clearly better only for a larger budget in the ellipsoid case.
    contextual_budget = budget if ellipsoid else 3 * budget
    contextual_budget *= int(max(1, np.sqrt(scale)))

    # Let us run the comparison.
    recommendations: tp.List[np.ndarray] = []
    for name in ("MetaModel", "CMA" if dimension > 1 else "OnePlusOne"):
        opt = registry[name](dimension, contextual_budget, num_workers=num_workers)
        recommendations.append(opt.minimize(_target).value)
    metamodel_recom, default_recom = recommendations  # pylint: disable=unbalanced-tuple-unpacking

    # Let us assert that MetaModel is better.
    assert _target(default_recom) > _target(metamodel_recom)

    # With large budget, the difference should be significant.
    if budget > 60 * dimension:
        assert _target(default_recom) > 4.0 * _target(metamodel_recom)

    # ... even more in the non ellipsoid case.
    if budget > 60 * dimension and not ellipsoid:
        assert _target(default_recom) > 7.0 * _target(metamodel_recom)


@pytest.mark.parametrize(  # type: ignore
    "penalization,expected,as_layer",
    [
        (False, [1.005573e00, 3.965783e-04], False),
        (True, [0.999975, -0.111235], False),
        (False, [1.000760, -5.116619e-4], True),
    ],
)
@testing.suppress_nevergrad_warnings()  # hides failed constraints
def test_constrained_optimization(penalization: bool, expected: tp.List[float], as_layer: bool) -> None:
    def constraint(i: tp.Any) -> tp.Union[bool, float]:
        if penalization:
            return -float(abs(i[1]["x"][0] - 1))
        out = i[1]["x"][0] >= 1
        return out if not as_layer else float(not out)

    parametrization = ng.p.Instrumentation(x=ng.p.Array(shape=(1,)), y=ng.p.Scalar())
    optimizer = optlib.OnePlusOne(parametrization, budget=100)
    optimizer.parametrization.random_state.seed(12)
    if penalization:
        optimizer._constraints_manager.update(max_trials=10, penalty_factor=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.parametrization.register_cheap_constraint(constraint, as_layer=as_layer)
    recom = optimizer.minimize(_square, verbosity=2)
    np.testing.assert_array_almost_equal([recom.kwargs["x"][0], recom.kwargs["y"]], expected)


@pytest.mark.parametrize("name", registry)  # type: ignore
def test_parametrization_offset(name: str) -> None:
    if "PSO" in name or "BO" in name:
        raise SkipTest("PSO and BO have large initial variance")
    if "Cobyla" in name and platform.system() == "Windows":
        raise SkipTest("Cobyla is flaky on Windows for unknown reasons")
    parametrization = ng.p.Instrumentation(ng.p.Array(init=[1e12, 1e12]))
    with testing.suppress_nevergrad_warnings():
        optimizer = registry[name](parametrization, budget=100, num_workers=1)
    for k in range(10 if "BO" not in name else 2):
        candidate = optimizer.ask()
        assert (
            candidate.args[0][0] > 100
        ), f"Candidate value[0] at iteration #{k} is below 100: {candidate.value}"
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
    [
        ("Shiwa", 1, 10, 1, "Cobyla"),
        ("Shiwa", 1, 10, 2, "OnePlusOne"),
        (
            "Shiwa",
            ng.p.Log(lower=1, upper=1000).set_integer_casting(),
            10,
            2,
            "DoubleFastGADiscreteOnePlusOne",
        ),
        ("NGOpt", 1, 10, 1, "MetaModel"),
        ("NGOpt", 1, 10, 2, "MetaModel"),
        (
            "NGOpt",
            ng.p.Log(lower=1, upper=1000).set_integer_casting(),
            10,
            2,
            "DoubleFastGADiscreteOnePlusOne",
        ),
        ("NGOpt8", ng.p.TransitionChoice(range(30), repetitions=10), 10, 2, "CMandAS2"),
        ("NGOpt8", ng.p.TransitionChoice(range(3), repetitions=10), 10, 2, "AdaptiveDiscreteOnePlusOne"),
        ("NGOpt", ng.p.TransitionChoice(range(30), repetitions=10), 10, 2, "DiscreteLenglerOnePlusOne"),
        ("NGOpt", ng.p.TransitionChoice(range(3), repetitions=10), 10, 2, "DiscreteLenglerOnePlusOne"),
        ("NGO", 1, 10, 1, "Cobyla"),
        ("NGO", 1, 10, 2, "OnePlusOne"),
    ],  # pylint: disable=too-many-arguments
)
@testing.suppress_nevergrad_warnings()
def test_ngopt_selection(
    name: str, param: tp.Any, budget: int, num_workers: int, expected: str, caplog: tp.Any
) -> None:
    with caplog.at_level(logging.DEBUG, logger="nevergrad.optimization.optimizerlib"):
        # pylint: disable=expression-not-assigned
        optlib.registry[name](param, budget=budget, num_workers=num_workers).optim  # type: ignore
        pattern = rf".*{name} selected (?P<name>\w+?) optimizer\."
        match = re.match(pattern, caplog.text.splitlines()[-1])
        assert match is not None, f"Did not detect selection in logs: {caplog.text}"
        assert match.group("name") == expected


def test_bo_ordering() -> None:
    with testing.suppress_nevergrad_warnings():  # tests do not need to be efficient
        optim = ng.optimizers.ParametrizedBO(initialization="Hammersley")(
            parametrization=ng.p.Choice(range(12)), budget=10
        )
    cand = optim.ask()
    optim.tell(cand, 12)
    optim.provide_recommendation()


@skip_win_perf  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "name,dimension,num_workers,fake_learning,budget,expected",
    [
        ("NGOpt8", 3, 1, False, 100, ["OnePlusOne", "OnePlusOne"]),
        ("NGOpt8", 3, 1, False, 200, ["SQP", "SQP"]),
        ("NGOpt8", 3, 1, True, 1000, ["SQP", "monovariate", "monovariate"]),
        (None, 3, 1, False, 1000, ["CMA", "OnePlusOne"]),
        (None, 3, 20, False, 1000, ["MetaModel", "OnePlusOne"]),
    ],
)
def test_ngo_split_optimizer(
    name: tp.Optional[str],
    dimension: int,
    num_workers: int,
    fake_learning: bool,
    budget: int,
    expected: tp.List[str],
) -> None:
    param: ng.p.Parameter = ng.p.Instrumentation(
        ng.p.Instrumentation(
            # a log-distributed scalar between 0.001 and 1.0
            learning_rate=ng.p.Log(lower=0.001, upper=1.0),
            # an integer from 1 to 12
            batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting(),
            # either "conv" or "fc"
            architecture=ng.p.Choice(["conv", "fc"]),
        )
        if fake_learning
        else ng.p.Choice(["const", ng.p.Array(init=list(range(dimension)))])
    )
    opt: base.OptCls = (
        xpvariants.MetaNGOpt10
        if name is None
        else (optlib.ConfSplitOptimizer(multivariate_optimizer=optlib.registry[name]))
    )
    optimizer = opt(param, budget=budget, num_workers=num_workers)
    names = [o.optim.name if o.dimension != 1 or name is None else "monovariate" for o in optimizer.optims]  # type: ignore
    assert names == expected


@skip_win_perf  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "budget,with_int",
    [
        (150, True),
        (200, True),
        (666, True),
        (2000, True),
        (66, False),
        (200, False),
        (666, False),
        (2000, False),
    ],
)
def test_ngopt_on_simple_realistic_scenario(budget: int, with_int: bool) -> None:
    def fake_training(learning_rate: float, batch_size: int, architecture: str) -> float:
        # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
        return (learning_rate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if architecture == "conv" else 10)

    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        # a log-distributed scalar between 0.001 and 1.0
        learning_rate=ng.p.Log(lower=0.001, upper=1.0),
        # an integer from 1 to 12
        batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting()
        if with_int
        else ng.p.Scalar(lower=1, upper=12),
        # either "conv" or "fc"
        architecture=ng.p.Choice(["conv", "fc"]),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    recommendation = optimizer.minimize(fake_training)
    result = fake_training(**recommendation.kwargs)
    assert result < 1.0006 if with_int else 5e-3, f"{result} not < {1.0006 if with_int else 5e-3}"


def _multiobjective(z: np.ndarray) -> tp.Tuple[float, float, float]:
    x, y = z
    return (abs(x - 1), abs(y + 1), abs(x - y))


@pytest.mark.parametrize("name", ["DE", "ES", "OnePlusOne"])  # type: ignore
@testing.suppress_nevergrad_warnings()  # hides bad loss
def test_mo_constrained(name: str) -> None:
    optimizer = optlib.registry[name](2, budget=60)
    optimizer.parametrization.random_state.seed(12)

    def constraint(arg: tp.Any) -> bool:  # pylint: disable=unused-argument
        """Random constraint to mess up with the optimizer"""
        return bool(optimizer.parametrization.random_state.rand() > 0.8)

    optimizer.parametrization.register_cheap_constraint(constraint)
    optimizer.minimize(_multiobjective)
    point = optimizer.parametrization.spawn_child(new_value=np.array([1.0, 1.0]))  # on the pareto
    optimizer.tell(point, _multiobjective(point.value))
    if isinstance(optimizer, es._EvolutionStrategy):
        assert optimizer._rank_method is not None  # make sure the nsga2 ranker is used


@pytest.mark.parametrize("name", ["DE", "ES", "OnePlusOne"])  # type: ignore
@testing.suppress_nevergrad_warnings()  # hides bad loss
def test_mo_with_nan(name: str) -> None:
    param = ng.p.Instrumentation(x=ng.p.Scalar(lower=0, upper=5), y=ng.p.Scalar(lower=0, upper=3))
    optimizer = optlib.registry[name](param, budget=60)
    optimizer.tell(ng.p.MultiobjectiveReference(), [10, 10, 10])
    for _ in range(50):
        cand = optimizer.ask()
        optimizer.tell(cand, [-38, 0, np.nan])


@pytest.mark.parametrize("name", ["LhsDE", "RandomSearch"])  # type: ignore
def test_uniform_sampling(name: str) -> None:
    param = ng.p.Scalar(lower=-100, upper=100).set_mutation(sigma=1)
    opt = optlib.registry[name](param, budget=600, num_workers=100)
    above_50 = 0
    for _ in range(100):
        above_50 += abs(opt.ask().value) > 50
    assert above_50 > 20  # should be around 50


def test_paraportfolio_de() -> None:
    workers = 40
    opt = optlib.ParaPortfolio(12, budget=100 * workers, num_workers=workers)
    for _ in range(3):
        cands = [opt.ask() for _ in range(workers)]
        for cand in cands:
            opt.tell(cand, np.random.rand())


def test_cma_logs(capsys: tp.Any) -> None:
    opt = registry["CMA"](2, budget=300, num_workers=4)
    [opt.ask() for _ in range(4)]  # pylint: disable=expression-not-assigned
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
