# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# (C) Copyright 2020 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This work is implemented under the Formulations and Computational Engineering (FORCE) project within Horizon 2020
# (`NMBP-23-2016/721027 <https://www.the-force-project.eu>`_).
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from .hypervolume import (
    VectorNode,
    VectorLinkedList,
    HypervolumeIndicator,
)


def test_initialize_empty_node() -> None:
    dim = 4
    node = VectorNode(dim)

    assert isinstance(node.coordinates, np.ndarray)
    for entry in node.next:
        assert entry is node
    for entry in node.prev:
        assert entry is node

    assert list(node.area) == [0.0] * dim
    assert list(node.volume) == [0.0] * dim
    assert str(node) == "None"


def test_initialize_node() -> None:
    dim = 4
    coordinates = [1.0, 2.0, 3.0]
    node = VectorNode(dim, coordinates=coordinates)

    assert isinstance(node.coordinates, np.ndarray)
    assert list(node.coordinates) == coordinates
    for entry in node.next:
        assert entry is node
    for entry in node.prev:
        assert entry is node

    assert list(node.area) == [0.0] * dim
    assert list(node.volume) == [0.0] * dim
    assert str(node) == "[1. 2. 3.]"

    node.configure_area(0)
    assert node.area[0] == 1.0
    assert node.area[1] == 0.0
    assert node.area[2] == 0.0

    node.configure_area(1)
    assert node.area[0] == 1.0
    assert node.area[1] == -1.0
    assert node.area[2] == 0.0

    node.configure_area(2)
    assert node.area[0] == 1.0
    assert node.area[1] == -1.0
    assert node.area[2] == 2.0


def test_initialize_linked_list() -> None:
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    assert dim == multilist.dimension
    assert isinstance(multilist.sentinel, VectorNode)
    assert len(multilist.sentinel.prev) == 4
    assert len(multilist.sentinel.next) == 4
    assert len(multilist) == 4

    for d in range(dim):
        assert multilist.sentinel is multilist.sentinel.next[d]
        assert multilist.sentinel is multilist.sentinel.prev[d]

    assert len(multilist.sentinel.next) == len(multilist.sentinel.prev)
    assert len(multilist.sentinel.next) == len(multilist.sentinel.next[0].next)

    assert str(multilist) == "\n".join([str([])] * dim)


def test_append() -> None:
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    multilist.append(new_node, 0)

    for i in range(1, dim):
        assert new_node.next[i] is new_node
        assert new_node.prev[i] is new_node
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel

    assert new_node.next[0] is multilist.sentinel
    assert new_node.prev[0] is multilist.sentinel
    assert multilist.sentinel.next[0] is new_node
    assert multilist.sentinel.prev[0] is new_node

    another_node = VectorNode(dim)
    multilist.append(another_node, 0)
    for i in range(1, dim):
        assert new_node.next[i] is new_node
        assert new_node.prev[i] is new_node
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel

    assert new_node.next[0] is another_node
    assert new_node.prev[0] is multilist.sentinel
    assert multilist.sentinel.next[0] is new_node
    assert multilist.sentinel.prev[0] is another_node


def test_extend() -> None:
    dim = 1
    multilist = VectorLinkedList(dimension=dim)
    another_multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    another_node = VectorNode(dim)

    multilist.append(new_node, 0)
    multilist.append(another_node, 0)

    another_multilist.extend([new_node, another_node], 0)
    assert another_multilist.chain_length(0) == 2
    assert another_multilist.sentinel.next[0] is multilist.sentinel.next[0]
    assert another_multilist.sentinel.next[0].next[0] is multilist.sentinel.next[0].next[0]


def test_chain_length() -> None:
    dim = 3
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    multilist.append(new_node, 0)
    assert multilist.chain_length(0) == 1
    assert multilist.chain_length(1) == 0
    assert multilist.chain_length(2) == 0

    another_node = VectorNode(dim)
    multilist.append(another_node, 0)
    assert multilist.chain_length(0) == 2
    assert multilist.chain_length(1) == 0
    assert multilist.chain_length(2) == 0

    multilist.append(another_node, 1)
    assert multilist.chain_length(0) == 2
    assert multilist.chain_length(1) == 1
    assert multilist.chain_length(2) == 0

    multilist.append(new_node, 2)
    assert multilist.chain_length(0) == 2
    assert multilist.chain_length(1) == 1
    assert multilist.chain_length(2) == 1


def test_pop() -> None:
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    multilist.append(new_node, 0)

    popped_node = multilist.pop(new_node, 0 + 1)
    assert popped_node is new_node
    assert new_node.next[0] is multilist.sentinel
    assert new_node.prev[0] is multilist.sentinel
    for i in range(dim):
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel


def test_reinsert() -> None:
    dim = 2
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    another_node = VectorNode(dim)

    multilist.append(new_node, 0)
    multilist.append(another_node, 0)

    multilist.append(another_node, 1)
    multilist.append(new_node, 1)

    popped_node = multilist.pop(new_node, 1 + 1)

    multilist.reinsert(new_node, 0 + 1)
    assert multilist.chain_length(0) == 2
    assert multilist.chain_length(1) == 1
    assert new_node.next[0] is another_node
    assert new_node.prev[0] is multilist.sentinel
    assert another_node.prev[0] is new_node
    assert another_node.next[0] is multilist.sentinel
    assert another_node.prev[1] is multilist.sentinel
    assert another_node.next[1] is multilist.sentinel

    multilist.reinsert(popped_node, 1 + 1)
    assert multilist.chain_length(0) == 2
    assert multilist.chain_length(1) == 2
    assert another_node.prev[1] is multilist.sentinel
    assert another_node.next[1] is new_node
    assert new_node.prev[1] is another_node
    assert new_node.next[1] is multilist.sentinel


def test_iterate() -> None:
    dim = 1
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    another_node = VectorNode(dim)

    multilist.append(new_node, 0)
    multilist.append(another_node, 0)
    gen = multilist.iterate(0)
    assert next(gen) is new_node
    assert next(gen) is another_node

    yet_another_node = VectorNode(dim)
    multilist.append(yet_another_node, 0)
    gen = multilist.iterate(0, start=another_node)
    assert next(gen) is another_node
    assert next(gen) is yet_another_node


def test_reverse_iterate() -> None:
    dim = 1
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    another_node = VectorNode(dim)
    yet_another_node = VectorNode(dim)

    multilist.append(new_node, 0)
    multilist.append(another_node, 0)
    multilist.append(yet_another_node, 0)

    gen = multilist.reverse_iterate(0)
    assert next(gen) is yet_another_node
    assert next(gen) is another_node
    assert next(gen) is new_node

    gen = multilist.reverse_iterate(0, start=another_node)
    assert next(gen) is another_node
    assert next(gen) is new_node


def test_update_coordinate_bounds() -> None:
    bounds = np.array([-1.0, -1.0, -1.0])
    node = VectorNode(3, coordinates=[1.0, -2.0, -1.0])
    bounds = VectorLinkedList.update_coordinate_bounds(bounds, node, 0 + 1)
    assert list(bounds) == [-1, -1, -1]
    bounds = VectorLinkedList.update_coordinate_bounds(bounds, node, 1 + 1)
    assert list(bounds) == [-1, -2, -1]
    bounds = VectorLinkedList.update_coordinate_bounds(bounds, node, 2 + 1)
    assert list(bounds) == [-1, -2, -1]


def test_sort_by_index() -> None:
    nodes = [VectorNode(3, [1, 2, 3]), VectorNode(3, [2, 3, 1]), VectorNode(3, [3, 1, 2])]
    new_nodes = VectorLinkedList.sort_by_index(nodes, 0)
    assert new_nodes == nodes

    new_nodes = VectorLinkedList.sort_by_index(nodes, 1)
    assert new_nodes == [nodes[2], nodes[0], nodes[1]]

    new_nodes = VectorLinkedList.sort_by_index(nodes, 2)
    assert new_nodes == [nodes[1], nodes[2], nodes[0]]


def test_create_sorted() -> None:
    dimension = 3
    coordinates = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    linked_list = VectorLinkedList.create_sorted(dimension, coordinates)
    assert isinstance(linked_list, VectorLinkedList)
    assert list(linked_list.sentinel.next[0].coordinates) == [1, 2, 3]
    assert list(linked_list.sentinel.next[1].coordinates) == [3, 1, 2]
    assert list(linked_list.sentinel.next[2].coordinates) == [2, 3, 1]

    assert list(linked_list.sentinel.next[0].next[0].coordinates) == [2, 3, 1]
    assert list(linked_list.sentinel.next[1].next[1].coordinates) == [1, 2, 3]
    assert list(linked_list.sentinel.next[2].next[2].coordinates) == [3, 1, 2]


def test_version_consistency() -> None:
    reference = np.array([79, 89, 99])
    hv = HypervolumeIndicator(reference)
    front = np.array(
        [(110, 110, 100), (110, 90, 87), (80, 80, 36), (50, 50, 55), (105, 30, 43), (110, 110, 100)]
    )
    volume = hv.compute(front)
    assert volume == 11113.0


def test_reference_no_pointy() -> None:
    reference = np.array([10, 10])
    hv = HypervolumeIndicator(reference)
    front = np.array(
        [
            (11, 9),
            (9, 11),
        ]
    )
    volume = hv.compute(front)
    assert volume == -3  # not sure this is expected
