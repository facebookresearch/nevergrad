# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Native direct search optimizers: DSbase, DSproba, DSsubspace and DS3p."""

import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter


class _DSBase(base.Optimizer):
    """Common machinery of the native direct search methods DSbase, DSproba, DSsubspace and DS3p.

    The optimizer keeps an incumbent x (in standardized space) and its loss f(x), and polls points
    x + step * d for directions d provided by the subclass (_poll_directions), which also defines how the
    results of the polls update the incumbent and the step size (_tell_poll).
    The first point evaluated is the center of the parametrization, unless points were already told,
    in which case the search starts from the best of them (eg: when chained).
    Polls are tagged with the iteration they were asked at, so that the optimizer works asynchronously
    (num_workers > 1): results of polls from previous iterations ("stale" polls) and tell-not-asked points
    replace the incumbent if they improve on it, which starts a new iteration around the new incumbent
    without changing the step size.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initial_step: float = 1.0,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if initial_step <= 0:
            raise ValueError(f"initial_step must be positive, got {initial_step}")
        self._initial_step = initial_step
        self._step = initial_step
        self._x: tp.Optional[np.ndarray] = None  # incumbent, in standardized space
        self._fx = float("inf")  # loss of the incumbent
        self._iteration = 0  # changes whenever the incumbent or the step size changes
        self._num_polls = 0  # number of polls asked during the current iteration
        self._directions: tp.List[np.ndarray] = []  # directions still to be polled in this iteration
        self._pending: tp.Dict[str, int] = {}  # candidate uid -> iteration it was asked at (-1 for x0)

    def _new_iteration(self) -> None:
        self._iteration += 1
        self._num_polls = 0
        self._directions = []

    def _poll_directions(self) -> tp.List[np.ndarray]:
        """Returns the next directions to poll during the current iteration
        (called whenever all the previously provided directions have been asked)
        """
        raise NotImplementedError

    def _tell_poll(self, data: np.ndarray, loss: tp.FloatLoss) -> None:
        """Updates the incumbent and step size with the loss of a poll of the current iteration"""
        raise NotImplementedError

    def _random_unit_vector(self) -> np.ndarray:
        """Direction drawn uniformly on the unit sphere"""
        d = self._rng.normal(size=self.dimension)
        return d / max(np.linalg.norm(d), 1e-300)

    def _internal_ask_candidate(self) -> p.Parameter:
        if self._x is None:
            best = self.current_bests["average"]
            loss = best.get_estimation("average")
            if np.isfinite(loss):  # start from the best point told so far (eg: when chained)
                self._x = np.array(best.x, copy=True)
                self._fx = loss
            else:  # evaluate the initial point first
                self._x = np.zeros(self.dimension)
                candidate = self.parametrization.spawn_child()
                self._pending[candidate.uid] = -1
                return candidate
        if not self._directions:
            self._directions = self._poll_directions()
        data = self._x + self._step * self._directions.pop(0)
        self._num_polls += 1
        candidate = self.parametrization.spawn_child().set_standardized_data(data)
        self._pending[candidate.uid] = self._iteration
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        iteration = self._pending.pop(candidate.uid, None)
        if iteration is None:  # eg: a candidate which was rejected by constraints before being asked
            self._internal_tell_not_asked(candidate, loss)
            return
        data = candidate.get_standardized_data(reference=self.parametrization)
        if iteration != self._iteration:  # initial point or stale poll (num_workers > 1)
            self._update_if_better(data, loss)
            return
        self._tell_poll(data, loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        if self._x is None:
            return  # the first ask will start from the best point told so far
        self._update_if_better(candidate.get_standardized_data(reference=self.parametrization), loss)

    def _update_if_better(self, data: np.ndarray, loss: tp.FloatLoss) -> None:
        """Moves to a point which was not polled at the current iteration if it improves
        on the incumbent, without changing the step size."""
        assert self._x is not None
        if loss < self._fx:
            self._fx = loss
            if not np.array_equal(data, self._x):
                self._x = np.array(data, copy=True)
                self._new_iteration()


class _SufficientDecreaseDS(_DSBase):
    """Direct search with sufficient decrease and an adaptive step size, the poll set being defined by
    subclasses (see Kolda, Lewis, Torczon, "Optimization by direct search: new perspectives on some classical
    and modern methods", SIAM Review 45(3), 2003, and Gratton, Royer, Vicente, Zhang, "Direct search based
    on probabilistic descent", SIAM J. Optim. 25(3), 2015).

    Each iteration k polls x_k + alpha_k * d for the directions d of the poll set.
    A trial point is accepted only if it yields sufficient decrease f(trial) < f(x_k) - c * alpha_k ** 2.
    Polling is opportunistic: the iteration stops at the first accepted trial point, and the step size is
    then expanded (alpha *= expansion). If the _poll_size polls of the iteration all fail, the step size is
    contracted (alpha *= contraction).
    When a contraction takes the step size below min_step, the search has converged: the step size is reset
    to initial_step (restart from the incumbent) so that the remaining budget is not wasted.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, initial_step=initial_step)
        if expansion < 1:
            raise ValueError(f"expansion must be at least 1, got {expansion}")
        if not 0 < contraction < 1:
            raise ValueError(f"contraction must be in (0, 1), got {contraction}")
        if forcing_constant < 0:
            raise ValueError(f"forcing_constant must be non-negative, got {forcing_constant}")
        if not 0 <= min_step < initial_step:
            raise ValueError(f"min_step must be in [0, initial_step), got {min_step}")
        self._expansion = expansion
        self._contraction = contraction
        self._forcing_constant = forcing_constant
        self._min_step = min_step
        self._failures = 0  # number of failed polls during the current iteration
        self._poll_size = 2  # number of failed polls after which the iteration is unsuccessful

    def _new_iteration(self) -> None:
        super()._new_iteration()
        self._failures = 0

    def _tell_poll(self, data: np.ndarray, loss: tp.FloatLoss) -> None:
        if loss < self._fx - self._forcing_constant * self._step**2:  # sufficient decrease
            self._x = data
            self._fx = loss
            self._step *= self._expansion
            self._new_iteration()
        else:
            self._failures += 1
            if self._failures >= self._poll_size:
                self._step *= self._contraction
                if self._step < self._min_step:  # converged: restart with the initial step size
                    self._step = self._initial_step
                self._new_iteration()


class _CoordinateDS(_SufficientDecreaseDS):
    """Deterministic direct search polling the coordinate directions (compass search), with
    sufficient decrease and an adaptive step size.

    Native (MIT) implementation written from the paper:
    T. G. Kolda, R. M. Lewis, V. Torczon, "Optimization by direct search: new perspectives on some
    classical and modern methods", SIAM Review 45(3), 2003.

    Each iteration polls x + alpha * d for d in the positive spanning set {e_1, -e_1, ..., e_n, -e_n},
    in this order (see _SufficientDecreaseDS for the update rule).
    With num_workers = 1, the method is deterministic.
    With num_workers > 1, the polls of an iteration are evaluated concurrently. When all 2n coordinate
    directions of the iteration have been asked but some results are still pending, additional pairs
    (d, -d) of random directions (d uniform on the unit sphere) are polled so that all workers are used,
    as allowed by the generating set search framework of the paper (additional search directions).
    The iteration is unsuccessful once max(2n, 2 * ceil(num_workers / 2)) polls have failed.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
    ) -> None:
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            initial_step=initial_step,
            expansion=expansion,
            contraction=contraction,
            forcing_constant=forcing_constant,
            min_step=min_step,
        )
        self._poll_size = max(2 * self.dimension, 2 * ((self.num_workers + 1) // 2))

    def _poll_directions(self) -> tp.List[np.ndarray]:
        if self._num_polls == 0:  # beginning of the iteration: the coordinate directions
            eye = np.eye(self.dimension)
            return [sign * eye[i] for i in range(self.dimension) for sign in (1.0, -1.0)]
        d = self._random_unit_vector()  # additional directions, for num_workers > 2n
        return [d, -d]


class _ProbabilisticDS(_SufficientDecreaseDS):
    """Direct search based on probabilistic descent, polling the positive and negative directions of
    an orthonormal basis of a random subspace.

    Native (MIT) implementation written from the papers:
    - S. Gratton, C. W. Royer, L. N. Vicente, Z. Zhang, "Direct search based on probabilistic descent",
      SIAM J. Optim. 25(3), 2015 (subspace_dimension=1);
    - L. Roberts, C. W. Royer, "Direct search based on probabilistic descent in reduced spaces",
      SIAM J. Optim. 33(4), 2023 (subspace_dimension > 1).

    Each iteration draws a uniformly distributed (Haar) random subspace of dimension p = subspace_dimension,
    with orthonormal basis q_1, ..., q_p (the columns of the orthogonal factor of a Gaussian n x p matrix),
    and polls x + alpha * d for d in {q_1, -q_1, ..., q_p, -q_p} (see _SufficientDecreaseDS for the update
    rule).
    For p = 1, this means polling d and -d with d uniform on the unit sphere.
    With num_workers > 1, each iteration polls the bases of ceil(num_workers / (2p)) independent random
    subspaces concurrently, and is unsuccessful once all of these polls failed.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
        subspace_dimension: int = 1,
    ) -> None:
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            initial_step=initial_step,
            expansion=expansion,
            contraction=contraction,
            forcing_constant=forcing_constant,
            min_step=min_step,
        )
        if subspace_dimension < 1:
            raise ValueError(f"subspace_dimension must be at least 1, got {subspace_dimension}")
        self._subspace_dimension = min(subspace_dimension, self.dimension)
        num_dirs = 2 * self._subspace_dimension  # directions polled for each random subspace
        self._poll_size = num_dirs * ((self.num_workers + num_dirs - 1) // num_dirs)

    def _poll_directions(self) -> tp.List[np.ndarray]:
        if self._subspace_dimension == 1:
            d = self._random_unit_vector()
            return [d, -d]
        gaussian = self._rng.normal(size=(self.dimension, self._subspace_dimension))
        q, r = np.linalg.qr(gaussian)
        q *= np.where(np.diag(r) < 0, -1.0, 1.0)  # sign correction, for a uniformly distributed basis
        return [sign * q[:, i] for i in range(self._subspace_dimension) for sign in (1.0, -1.0)]


class _StochasticThreePoints(_DSBase):
    """Stochastic Three Points method: keeps the best of x, x + alpha_k * s and x - alpha_k * s.

    Native (MIT) implementation written from the paper:
    E. H. Bergou, E. Gorbunov, P. Richtarik, "Stochastic three points method for unconstrained smooth
    minimization", SIAM J. Optim. 30(4), 2020.

    Each iteration k draws s_k uniformly on the unit sphere, evaluates x_k + alpha_k * s_k and
    x_k - alpha_k * s_k, and sets x_{k+1} to the best of these two points and x_k (whose loss is not
    re-evaluated). The step size follows the decreasing schedule alpha_k = initial_step / sqrt(k + 1),
    k = 0, 1, 2... being the number of completed iterations.
    With num_workers > 1, each iteration evaluates ceil(num_workers / 2) independent pairs
    (x_k + alpha_k * s, x_k - alpha_k * s) concurrently, and keeps the best of all of them and x_k
    (this is the paper's method for num_workers <= 2).
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initial_step: float = 1.0,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, initial_step=initial_step)
        self._poll_size = 2 * ((self.num_workers + 1) // 2)  # number of polls of each iteration
        self._num_completed = 0  # number of completed iterations (k)
        self._num_told = 0  # number of polls of the current iteration which have been told
        self._best: tp.Optional[tp.Tuple[np.ndarray, tp.FloatLoss]] = None  # best poll of the iteration

    def _new_iteration(self) -> None:
        super()._new_iteration()
        self._num_told = 0
        self._best = None

    def _poll_directions(self) -> tp.List[np.ndarray]:
        s = self._random_unit_vector()
        return [s, -s]

    def _tell_poll(self, data: np.ndarray, loss: tp.FloatLoss) -> None:
        self._num_told += 1
        if self._best is None or loss < self._best[1]:
            self._best = (data, loss)
        # the iteration is complete once poll_size of its polls have been told: results of additional polls
        # (eg: when asking more than num_workers candidates at once) will be handled as stale polls
        if self._num_told >= self._poll_size:
            if self._best[1] < self._fx:
                self._x, self._fx = self._best
            self._num_completed += 1
            self._step = self._initial_step / np.sqrt(self._num_completed + 1)
            self._new_iteration()


class ParametrizedDSbase(base.ConfiguredOptimizer):
    """Deterministic direct search polling the coordinate directions (compass search), with sufficient
    decrease and an adaptive step size (Kolda, Lewis, Torczon, "Optimization by direct search: new
    perspectives on some classical and modern methods", SIAM Review 45(3), 2003).

    At each iteration, the directions e_1, -e_1, ..., e_n, -e_n (a positive spanning set) are polled in turn
    around the incumbent with step size alpha. The first trial point which decreases the loss by at least
    forcing_constant * alpha ** 2 becomes the incumbent and the step size is multiplied by expansion;
    if all 2n polls fail, the step size is multiplied by contraction.

    Parameters
    ----------
    initial_step: float
        initial step size alpha_0, in the standardized space (where the parametrization has scale 1)
    expansion: float
        step size multiplier after a successful iteration (phi >= 1)
    contraction: float
        step size multiplier after an unsuccessful iteration (0 < theta < 1)
    forcing_constant: float
        constant c of the forcing function rho(alpha) = c * alpha ** 2 used in the sufficient decrease test
    min_step: float
        when a contraction takes the step size below this value, the search is considered converged and
        restarts from the best point found with step size initial_step, so that the remaining budget is
        still used for exploration (0 disables restarts)

    Note
    ----
    - Defaults: theta = 1/2 is the step halving of classical compass search. The paper's generating set
      search framework allows any bounded expansion phi >= 1 when a sufficient decrease condition is used;
      phi = 2 lets the step size grow back quickly after a success (classical compass search uses phi = 1).
      rho(alpha) = c * alpha ** 2 is a forcing function as required by the paper, with a small c = 1e-3,
      and alpha_0 = 1 matches the unit scale of the standardized space. These are the same defaults as for
      DSproba.
    - The optimizer is deterministic with num_workers = 1. It works asynchronously: with num_workers > 1,
      the polls of an iteration are evaluated concurrently, and when all 2n coordinate directions of the
      iteration have been asked while results are still pending, additional random directions (d, -d)
      are polled so that all workers are used. The step size is contracted once
      max(2n, 2 * ceil(num_workers / 2)) polls of the iteration have failed. Results of polls from previous
      iterations, and tell-not-asked points, replace the incumbent if they improve it, without changing
      the step size.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
    ) -> None:
        super().__init__(_CoordinateDS, locals())


DSbase = ParametrizedDSbase().set_name("DSbase", register=True)


class ParametrizedDSproba(base.ConfiguredOptimizer):
    """Direct search based on probabilistic descent (Gratton, Royer, Vicente, Zhang,
    "Direct search based on probabilistic descent", SIAM J. Optim. 25(3), 2015).

    At each iteration, two opposite directions d and -d (d uniform on the unit sphere) are polled
    around the incumbent with step size alpha. A trial point is accepted only if it decreases the
    loss by at least forcing_constant * alpha ** 2; the step size is then multiplied by expansion,
    otherwise it is multiplied by contraction once all polled directions failed.

    Parameters
    ----------
    initial_step: float
        initial step size alpha_0, in the standardized space (where the parametrization has scale 1)
    expansion: float
        step size multiplier after a successful iteration (gamma >= 1)
    contraction: float
        step size multiplier after an unsuccessful iteration (0 < theta < 1)
    forcing_constant: float
        constant c of the forcing function rho(alpha) = c * alpha ** 2 used in the sufficient decrease test
    min_step: float
        when a contraction takes the step size below this value, the search is considered converged and
        restarts from the best point found with step size initial_step, so that the remaining budget is
        still used for exploration (0 disables restarts)

    Note
    ----
    - Defaults: with gamma = 2 and theta = 1/2, the paper's convergence analysis requires the polling sets
      to be p0-probabilistically descent with p0 > ln(theta) / ln(theta / gamma) = 1/2. This holds for
      {d, -d}: one of the two directions always makes a non-negative angle with the negative gradient,
      so the probability that one of them has a cosine of at least kappa with it tends to 1 as kappa -> 0.
      c = 1e-3 is a small constant, as is usual for sufficient decrease tests, and alpha_0 = 1 matches the
      unit scale of the standardized space.
    - The optimizer works asynchronously: with num_workers > 1, each iteration polls
      ceil(num_workers / 2) independent pairs (d, -d) concurrently, and contracts the step size
      only when all of them failed. Results of polls from previous iterations, and tell-not-asked
      points, replace the incumbent if they improve it, without changing the step size.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
    ) -> None:
        super().__init__(_ProbabilisticDS, locals())


DSproba = ParametrizedDSproba().set_name("DSproba", register=True)


class ParametrizedDSsubspace(base.ConfiguredOptimizer):
    """Direct search based on probabilistic descent in reduced spaces (Roberts, Royer, "Direct search based
    on probabilistic descent in reduced spaces", SIAM J. Optim. 33(4), 2023).

    At each iteration, a uniformly distributed random subspace of dimension p = subspace_dimension is drawn,
    and the 2p directions q_1, -q_1, ..., q_p, -q_p, where q_1, ..., q_p is an orthonormal basis of the
    subspace (a random orthogonal sketch), are polled in turn around the incumbent with step size alpha.
    The first trial point which decreases the loss by at least forcing_constant * alpha ** 2 becomes the
    incumbent and the step size is multiplied by expansion; if all 2p polls fail, the step size is
    multiplied by contraction.

    Parameters
    ----------
    subspace_dimension: int
        dimension p of the random subspaces (capped to the dimension of the problem). p = 1 is DSproba.
    initial_step: float
        initial step size alpha_0, in the standardized space (where the parametrization has scale 1)
    expansion: float
        step size multiplier after a successful iteration (gamma >= 1)
    contraction: float
        step size multiplier after an unsuccessful iteration (0 < theta < 1)
    forcing_constant: float
        constant c of the forcing function rho(alpha) = c * alpha ** 2 used in the sufficient decrease test
    min_step: float
        when a contraction takes the step size below this value, the search is considered converged and
        restarts from the best point found with step size initial_step, so that the remaining budget is
        still used for exploration (0 disables restarts)

    Note
    ----
    - Defaults: the point of the method is that p can be chosen independently of (and much smaller than)
      the dimension n, so that an unsuccessful iteration costs 2p evaluations instead of 2n for DSbase.
      p = 2 is the smallest subspace dimension which differs from DSproba (p = 1), and it polls a
      positive spanning set of a plane, which increases the probability of finding a descent direction at
      each iteration for a cost which is still independent of n. With gamma = 2 and theta = 1/2, the
      polling sets must be probabilistically descent with probability above 1/2, as in DSproba.
      c = 1e-3 and alpha_0 = 1 are the same as for DSbase and DSproba.
    - The optimizer works asynchronously: with num_workers > 1, each iteration polls the bases of
      ceil(num_workers / (2p)) independent random subspaces concurrently, and contracts the step size
      only when all of these polls failed. Results of polls from previous iterations, and tell-not-asked
      points, replace the incumbent if they improve it, without changing the step size.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        subspace_dimension: int = 2,
        initial_step: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        forcing_constant: float = 1e-3,
        min_step: float = 1e-8,
    ) -> None:
        super().__init__(_ProbabilisticDS, locals())


DSsubspace = ParametrizedDSsubspace().set_name("DSsubspace", register=True)


class ParametrizedDS3p(base.ConfiguredOptimizer):
    """Stochastic Three Points method (Bergou, Gorbunov, Richtarik, "Stochastic three points method for
    unconstrained smooth minimization", SIAM J. Optim. 30(4), 2020).

    At each iteration k, a direction s_k is drawn uniformly on the unit sphere, and the next iterate is
    the best of x_k, x_k + alpha_k * s_k and x_k - alpha_k * s_k, with step size
    alpha_k = initial_step / sqrt(k + 1).

    Parameters
    ----------
    initial_step: float
        initial step size alpha_0, in the standardized space (where the parametrization has scale 1)

    Note
    ----
    - Defaults: alpha_k = alpha_0 / sqrt(k + 1) is the paper's decreasing step size schedule for smooth
      nonconvex functions, which does not require knowing the budget nor any smoothness constant (the paper's
      other schedules do). Directions uniform on the unit sphere are one of the distributions analyzed in the
      paper, and alpha_0 = 1 matches the unit scale of the standardized space.
    - The step size only depends on the number of iterations, so the precision of the result is limited
      by the budget: after K iterations (2K evaluations), the step size is alpha_0 / sqrt(K + 1).
    - With num_workers > 1, each iteration evaluates ceil(num_workers / 2) independent pairs of opposite
      points concurrently and keeps the best of all of them and x_k. Results of polls from previous
      iterations, and tell-not-asked points, replace the incumbent if they improve it, without changing
      the step size.
    """

    # pylint: disable=unused-argument
    def __init__(self, *, initial_step: float = 1.0) -> None:
        super().__init__(_StochasticThreePoints, locals())


DS3p = ParametrizedDS3p().set_name("DS3p", register=True)
