# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
import networkx as nx
import cdt
from cdt.metrics import precision_recall, SHD
import nevergrad as ng
from ..base import ExperimentFunction


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
class CausalDiscovery(ExperimentFunction):
    """
    All parameters, except for the `generator` argument, are used to configure `acylicgraph`.
    For the setting of the acylic graph generator, please refer to https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/data.html

    Parameters
    ----------
    generator: str
        'tuebingen', 'sachs', 'dream4' or 'acylicgraph'
    causal_mechanism: str
        'linear', 'polynomial', 'sigmoid_add', 'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'
    noise: str or function
        Type of noise function to use in the generative process
    noise_coeff: float
        Proportion of noise
    npoints: int
        Total number of data points
    nodes: int
        Total number of nodes (i.e. variables)
    parents_max: int
        Maximum number of parent nodes
    expected_nedges: int
        Number of edge per node expected if erdos graph is used
    """

    def __init__(
        self,
        generator: str = "sachs",
        causal_mechanism: str = "linear",
        noise="gaussian",
        noise_coeff: float = 0.4,
        npoints: int = 500,
        nodes: int = 20,
        parents_max: int = 5,
        expected_degree: int = 3,
    ) -> None:
        # Currently, there are three datasets as described in https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/data.html
        _dataset = ["tuebingen", "sachs", "dream4"]
        assert generator in _dataset + ["acylicgraph"]
        assert causal_mechanism in [
            "linear",
            "polynomial",
            "sigmoid_add",
            "sigmoid_mix",
            "gp_add",
            "gp_mix",
            "nn",
        ]
        if generator in _dataset:
            self._data, self._ground_truth_graph = cdt.data.load_dataset(generator)
        else:
            _generator = cdt.data.AcyclicGraphGenerator(
                causal_mechanism=causal_mechanism,
                noise=noise,
                noise_coeff=noise_coeff,
                npoints=npoints,
                nodes=nodes,
                parents_max=parents_max,
                expected_degree=expected_degree,
            )
            self._data, self._ground_truth_graph = _generator.generate()

        self._nvars = self._data.shape[1]
        param_links = ng.p.Choice([-1, 0, 1], repetitions=self._nvars * (self._nvars - 1) // 2)
        instru = ng.p.Instrumentation(network_links=param_links).set_name("")
        super().__init__(self.objective, instru)

    def objective(self, network_links: tp.Tuple[int]) -> float:
        output_graph = self.choices_to_graph(network_links)
        score = self.graph_score(output_graph)
        return -score

    def graph_score(self, test_graph) -> float:
        pr_score, _ = precision_recall(self._ground_truth_graph, test_graph)
        shd_score = SHD(self._ground_truth_graph, test_graph)
        return float(pr_score - shd_score)  # Higher better

    def choices_to_graph(self, network_links):
        output_graph = nx.DiGraph()
        k = 0
        for i in range(1, self._nvars):
            for j in range(i + 1, self._nvars):
                if network_links[k] == -1:
                    output_graph.add_edge(j, i)
                elif network_links[k] == +1:
                    output_graph.add_edge(i, j)
                k += 1
        return output_graph
