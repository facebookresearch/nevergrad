# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p

logger = logging.getLogger(__name__)


class CrowdingDistance:
    """This class implements the calculation of crowding distance for NSGA-II."""

    def accumulate_distance_per_objective(self, front: tp.List[p.Parameter], i: int):
        if isinstance(front[0].losses, np.ndarray) and front[0].losses.shape != ():
            is_multiobj: bool = len(front[0].losses) > 1  # isinstance(front[0].loss, np.ndarray)
        else:
            is_multiobj = False
        assert (not is_multiobj and (i == 0)) or is_multiobj

        # Sort the population by objective i
        if is_multiobj:
            front = sorted(front, key=lambda x: x.losses[i])
            objective_minn = front[0].losses[i]
            objective_maxn = front[-1].losses[i]
            assert objective_minn <= objective_maxn

            # Set the crowding distance
            front[0]._meta["crowding_distance"] = float("inf")
            front[-1]._meta["crowding_distance"] = float("inf")

            # All other intermediate solutions are assigned a distance value equal
            # to the absolute normalized difference in the function values of two
            # adjacent solutions.
            for j in range(1, len(front) - 1):
                distance = front[j + 1].losses[i] - front[j - 1].losses[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass  # undefined
                else:
                    distance = distance / float(objective_maxn - objective_minn)
                logger.debug("front[j]: %s distance: %s", front[j].uid, distance)
                # The overall crowding-distance value is calculated as the sum of
                # individual distance values corresponding to each objective.
                front[j]._meta["crowding_distance"] += distance
        else:
            front = sorted(front, key=lambda x: x.loss)  # type: ignore
            objective_minn = front[0].loss
            objective_maxn = front[-1].loss
            assert objective_minn is not None and objective_maxn is not None
            assert objective_minn <= objective_maxn

            # Set the crowding distance
            front[0]._meta["crowding_distance"] = float("inf")
            front[-1]._meta["crowding_distance"] = float("inf")

            # All other intermediate solutions are assigned a distance value equal
            # to the absolute normalized difference in the function values of two
            # adjacent solutions.
            for j in range(1, len(front) - 1):
                loss_next = front[j + 1].loss
                loss_prev = front[j - 1].loss
                if loss_next is None or loss_prev is None:
                    distance = 0
                else:
                    distance = loss_next - loss_prev

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass  # undefined
                else:
                    distance = distance / float(objective_maxn - objective_minn)
                # The overall crowding-distance value is calculated as the sum of
                # individual distance values corresponding to each objective.
                front[j]._meta["crowding_distance"] += distance

    def compute_distance(self, front: tp.List[p.Parameter]):
        """This function assigns the crowding distance to the solutions.
        :param front: The list of solutions.
        """
        size = len(front)

        if size == 0:
            return
        # The boundary solutions (solutions with smallest and largest function values)
        # are set to an infinite (maximum) distance value
        if size == 1:
            front[0]._meta["crowding_distance"] = float("inf")
            return
        if size == 2:
            front[0]._meta["crowding_distance"] = float("inf")
            front[1]._meta["crowding_distance"] = float("inf")
            return

        for f in front:
            f._meta["crowding_distance"] = 0.0

        if isinstance(front[0].losses, np.ndarray) and front[0].losses.shape != ():
            number_of_objectives = len(front[0].losses)
        else:
            number_of_objectives = 1

        for i in range(number_of_objectives):
            self.accumulate_distance_per_objective(front, i)

    def sort(self, candidates: tp.List[p.Parameter], in_place: bool = True) -> tp.List[p.Parameter]:
        if in_place:
            candidates.sort(
                key=lambda elem: elem._meta["crowding_distance"], reverse=True
            )  # Larger -> Less crowded
        return sorted(candidates, key=lambda elem: elem._meta["crowding_distance"], reverse=True)


class FastNonDominatedRanking:
    """Non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]"""

    def compare(self, candidate1: p.Parameter, candidate2: p.Parameter) -> int:
        """Compare the domainance relation of two candidates.

        :param candidate1: Candidate.
        :param candidate2: Candidate.
        """
        one_wins = np.sum(candidate1.losses < candidate2.losses)
        two_wins = np.sum(candidate2.losses < candidate1.losses)
        if one_wins > 0 and two_wins == 0:
            return -1
        if two_wins > 0 and one_wins == 0:
            return 1
        return 0

    def compute_ranking(
        self, candidates: tp.List[p.Parameter], k: tp.Optional[int] = None
    ) -> tp.List[tp.List[p.Parameter]]:
        """Compute ranking of candidates.

        :param candidates: List of candidates.
        :param k: Number of individuals.
        """
        for cand in candidates:
            cand._meta["non_dominated_rank"] = float("inf")
        if not candidates:
            return []
        losses = np.array([np.atleast_1d(c.losses) for c in candidates], dtype=float)
        # dominates[i, j]: candidate i is no worse than candidate j on every objective
        # and better on at least one (same relation as the compare method)
        no_worse = (losses[:, None] <= losses[None]).all(axis=2)
        better = (losses[:, None] < losses[None]).any(axis=2)
        dominates = np.asarray(no_worse & better)
        dominated_by_cnt = dominates.sum(axis=0)  # number of candidates dominating each candidate
        remaining = np.ones(len(candidates), dtype=bool)
        previous = np.zeros(0, dtype=int)  # indices of the previous front
        ranked_sublists: tp.List[tp.List[p.Parameter]] = []
        count = 0
        while remaining.any() and (k is None or count < k):
            front = np.flatnonzero(remaining & (dominated_by_cnt == 0))
            if previous.size:
                # order the front as the classic algorithm finds it, i.e. by the position of its last
                # dominator in the previous front, so that selection among ties is unchanged
                positions = np.where(dominates[previous][:, front], np.arange(previous.size)[:, None], -1)
                front = front[np.argsort(positions.max(axis=0), kind="stable")]
            remaining[front] = False
            dominated_by_cnt -= dominates[front].sum(axis=0)
            for i in front:
                candidates[i]._meta["non_dominated_rank"] = len(ranked_sublists)
            ranked_sublists.append([candidates[i] for i in front])
            count += front.size
            previous = front
        return ranked_sublists


def rank(
    population: tp.List[p.Parameter], n_selected: tp.Optional[int] = None
) -> tp.Dict[str, tp.Tuple[int, int, float]]:
    """implements the multi-objective ranking function of NSGA-II."""
    frontier_ranker = FastNonDominatedRanking()
    density_estimator = CrowdingDistance()
    selected_pop: tp.Dict[str, tp.Tuple[int, int, float]] = {}
    frontiers = frontier_ranker.compute_ranking(population)
    count = 0
    next_rank = 0
    for front_i, p_frontier in enumerate(frontiers):
        count += len(p_frontier)
        if n_selected is None or count > n_selected:
            density_estimator.compute_distance(p_frontier)
            density_estimator.sort(p_frontier)
            n_dist_calc = n_selected - len(selected_pop) if n_selected is not None else len(p_frontier)
            for c_i in range(0, n_dist_calc):
                selected_pop[p_frontier[c_i].uid] = (
                    next_rank,
                    front_i,
                    p_frontier[c_i]._meta["crowding_distance"],
                )
                next_rank += 1
            if n_selected is not None:
                break
        if n_selected is not None:
            for candidate in p_frontier:
                selected_pop[candidate.uid] = (next_rank, front_i, float("inf"))
            next_rank += 1
    return selected_pop
