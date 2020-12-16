# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import nevergrad as ng
from . import core


def test_causal_discovery_using_data_func() -> None:
    func = core.CausalDiscovery(generator="sachs")
    assert func._nvars == 11
    # Prepare recommendation
    param_links = ng.p.Choice([-1, 0, 1], repetitions=func._nvars * (func._nvars - 1) // 2)
    param_links.set_standardized_data([i % 2 for i in range(0, param_links.dimension)])
    result = func(network_links=param_links.value)
    assert np.isclose(17.425619834710744, result, atol=1e-10)


def test_causal_discovery_using_data_minimize() -> None:
    # Optimization should return the same result since the true graph is not random and small
    func = core.CausalDiscovery(generator="sachs")
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=500)
    optimizer.parametrization.random_state = np.random.RandomState(12)
    recommendation = optimizer.minimize(func)
    recommendation_score = func(**recommendation.kwargs)
    assert np.isclose(17.425619834710744, recommendation_score, atol=1e-10)  # Optimal
    assert len(recommendation.kwargs["network_links"]) == func._nvars * (func._nvars - 1) // 2


def test_causal_discovery_using_generator() -> None:
    nnodes = 13
    npoints = 55
    func = core.CausalDiscovery(generator="acylicgraph", npoints=npoints, nodes=nnodes)
    assert func._nvars == nnodes
    assert func._data.shape == (npoints, nnodes)
    assert func.graph_score(func._ground_truth_graph) == 1
