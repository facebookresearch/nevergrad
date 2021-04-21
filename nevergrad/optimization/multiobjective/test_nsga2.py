# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import nsga2

# pylint: disable=reimported,redefined-outer-name,unused-variable,unsubscriptable-object, unused-argument
# pylint: disable=import-outside-toplevel


def test_crowding_distance() -> None:
    params = ng.p.Tuple(ng.p.Scalar(lower=0, upper=2), ng.p.Scalar(lower=0, upper=2))

    candidates: tp.List[p.Parameter] = []
    v = sorted([random.uniform(0.01, 0.99) for i in range(4)])
    loss_values = [[0.0, 5.0], [v[0], v[3]], [v[1], v[2]], [1.0, 0.0]]
    for i, v in enumerate(loss_values):
        candidates.append(params.spawn_child().set_standardized_data(v))
        candidates[i]._losses = np.array(v)
    crowding_distance = nsga2.CrowdingDistance()
    crowding_distance.compute_distance(candidates)

    # For objective 1
    cdist_1 = (loss_values[2][0] - loss_values[0][0]) / abs(loss_values[0][0] - loss_values[-1][0])
    cdist_2 = (loss_values[3][0] - loss_values[1][0]) / abs(loss_values[0][0] - loss_values[-1][0])

    # For objective 2
    cdist_1 += (loss_values[0][1] - loss_values[2][1]) / abs(loss_values[0][1] - loss_values[-1][1])
    cdist_2 += (loss_values[1][1] - loss_values[3][1]) / abs(loss_values[0][1] - loss_values[-1][1])

    assert candidates[0]._meta["crowding_distance"] == float("inf")
    np.testing.assert_almost_equal(candidates[1]._meta["crowding_distance"], cdist_1, decimal=3)
    np.testing.assert_almost_equal(candidates[2]._meta["crowding_distance"], cdist_2, decimal=3)
    assert candidates[3]._meta["crowding_distance"] == float("inf")


def test_fast_non_dominated_ranking() -> None:
    params = ng.p.Tuple(ng.p.Scalar(lower=0, upper=2), ng.p.Scalar(lower=0, upper=2))

    loss_values = [[[0.0, 2.0], [1.0, 1.0]], [[0.0, 4.0], [1.0, 3.0], [3.0, 1.0]], [[2.0, 3.0], [4.0, 2.0]]]

    candidates: tp.List[p.Parameter] = []
    expected_frontiers = []
    for vals in loss_values:
        expected_frontier = []
        for v in vals:
            candidate = params.spawn_child().set_standardized_data(v)
            candidate._losses = np.array(v)
            candidates.append(candidate)
            expected_frontier.append(candidate)
        expected_frontiers.append(expected_frontier)

    ranking_method = nsga2.FastNonDominatedRanking()
    frontiers = ranking_method.compute_ranking(candidates)
    assert set(frontiers[0]) == set(expected_frontiers[0])
    assert set(frontiers[1]) == set(expected_frontiers[1])
    assert set(frontiers[2]) == set(expected_frontiers[2])


def get_nsga2_test_case_data():
    params = ng.p.Tuple(ng.p.Scalar(lower=0, upper=2), ng.p.Scalar(lower=0, upper=2))

    loss_values = [[[0.0, 2.0], [1.0, 1.0]], [[0.0, 4.0], [1.0, 3.0], [3.0, 1.0]], [[2.0, 3.0], [4.0, 2.0]]]

    candidates: tp.List[p.Parameter] = []
    expected_frontiers = []
    for vals in loss_values:
        expected_frontier = []
        for v in vals:
            candidate = params.spawn_child().set_standardized_data(v)
            candidate._losses = np.array(v)
            candidates.append(candidate)
            expected_frontier.append(candidate)
        expected_frontiers.append(expected_frontier)
    return candidates, expected_frontiers


def test_nsga2_ranking() -> None:
    candidates, expected_frontiers = get_nsga2_test_case_data()
    rank_result = nsga2.rank(candidates, len(candidates))

    assert len(rank_result) == len(candidates)
    for i, frontier in enumerate(expected_frontiers):
        for c in frontier:
            assert rank_result[c.uid][0] == i


def test_nsga2_ranking_2() -> None:
    candidates, expected_frontiers = get_nsga2_test_case_data()
    n_selected = len(expected_frontiers[0]) + len(expected_frontiers[1]) - 1
    rank_result = nsga2.rank(candidates, n_selected)

    assert len(rank_result) == n_selected
    # Check the first frontier
    max_rank_frontier1 = 0
    for c in expected_frontiers[0]:
        assert rank_result[c.uid][2] == float("inf")
        max_rank_frontier1 = max(max_rank_frontier1, rank_result[c.uid][0])

    # Check the second frontier
    n_cand_in_frontier2 = 0
    for c in expected_frontiers[1]:
        if c.uid in rank_result:
            n_cand_in_frontier2 += 1
            assert rank_result[c.uid][0] > max_rank_frontier1
    assert n_cand_in_frontier2 == len(expected_frontiers[1]) - 1


def test_nsga2_ranking_3() -> None:
    candidates, expected_frontiers = get_nsga2_test_case_data()
    rank_result = nsga2.rank(candidates, None)

    assert len(rank_result) == len(candidates)
    for i, frontier in enumerate(expected_frontiers):
        expect_n_non_inf = max(0, len(frontier) - 2)
        n_non_inf = 0
        for c in frontier:
            assert rank_result[c.uid][1] == i
            if rank_result[c.uid][2] != float("inf"):
                n_non_inf += 1
        assert n_non_inf == expect_n_non_inf


def test_nsga2_ranking_4():
    params = ng.p.Tuple(ng.p.Scalar(lower=0, upper=2))
    loss_values = [0.0, 1.0, -10.0, 1.0, 3.0, 1.0]
    candidates: tp.List[p.Parameter] = []
    for v in loss_values:
        candidate = params.spawn_child().set_standardized_data(v)
        candidate.loss = np.array(v)
        candidates.append(candidate)

    n_selected = 3
    rank_result = nsga2.rank(candidates, n_selected)
    candidates.sort(key=lambda x: rank_result[x.uid][0] if x.uid in rank_result else float("inf"))
    loss_from_rank = [r.loss for r in candidates[:n_selected]]
    loss_from_sorted = [np.array(v) for v in sorted(loss_values)[:n_selected]]
    assert loss_from_rank == loss_from_sorted
