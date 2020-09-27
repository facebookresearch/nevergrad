import warnings
import random
import numpy as np
import nevergrad as ng
from . import rankers

# pylint: disable=reimported,redefined-outer-name,unused-variable,unsubscriptable-object, unused-argument
# pylint: disable=import-outside-toplevel

def test_crowding_distance() -> None:
    params = ng.p.Tuple(
        ng.p.Scalar(lower=0, upper=2),
        ng.p.Scalar(lower=0, upper=2))

    candidates = []
    v = sorted([random.uniform(0.01, 0.99) for i in range(4)])
    loss_values = [[0.0, 5.0], [v[0], v[3]], [v[1], v[2]], [1.0, 0.0]]
    for i, v in enumerate(loss_values):
        candidates.append(params.spawn_child().set_standardized_data(v))
        candidates[i].loss = np.array(v)
    crowding_distance = rankers.CrowdingDistance()
    crowding_distance.compute_distance(candidates)

    # For objective 1
    cdist_1 = (loss_values[2][0] - loss_values[0][0]) / abs(loss_values[0][0] - loss_values[-1][0])
    cdist_2 = (loss_values[3][0] - loss_values[1][0]) / abs(loss_values[0][0] - loss_values[-1][0])

    # For objective 2
    cdist_1 += (loss_values[0][1] - loss_values[2][1]) / abs(loss_values[0][1] - loss_values[-1][1])
    cdist_2 += (loss_values[1][1] - loss_values[3][1]) / abs(loss_values[0][1] - loss_values[-1][1])

    assert candidates[0]._meta["crowding_distance"] == float('inf')
    np.testing.assert_almost_equal(candidates[1]._meta["crowding_distance"], cdist_1, decimal=3)
    np.testing.assert_almost_equal(candidates[2]._meta["crowding_distance"], cdist_2, decimal=3)
    assert candidates[3]._meta["crowding_distance"] == float('inf')


def test_fast_non_dominated_ranking() -> None:
    params = ng.p.Tuple(
        ng.p.Scalar(lower=0, upper=2),
        ng.p.Scalar(lower=0, upper=2))

    loss_values = [[[0.0, 2.0], [1.0, 1.0]], [[0.0, 4.0], [1.0, 3.0], [3.0, 1.0]], [[2.0, 3.0], [4.0, 2.0]]]

    candidates = {}
    expected_frontiers = []
    for vals in loss_values:
        expected_frontier = []
        for v in vals:
            candidate = params.spawn_child().set_standardized_data(v)
            candidate.loss = np.array(v)
            candidates[candidate.uid] = candidate
            expected_frontier.append(candidate)
        expected_frontiers.append(expected_frontier)

    ranking_method = rankers.FastNonDominatedRanking()
    frontiers = ranking_method.compute_ranking(candidates)
    assert set(frontiers[0]) == set(expected_frontiers[0])
    assert set(frontiers[1]) == set(expected_frontiers[1])
    assert set(frontiers[2]) == set(expected_frontiers[2])
    