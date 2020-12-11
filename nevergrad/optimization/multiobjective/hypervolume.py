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

import typing as tp
import numpy as np


class VectorNode:
    """A node object of the VectorLinkedList.
    A VectorNode is a point in a space with dim = `dimension`, and an optional
    `coordinate` (which can be assigned after the VectorNode initialization).
    The VectorNode object points to two arrays with self.next and self.prev attributes.
    The `self.next` contains a list of VectorNode (aka geometric points), such that
    `self.next[i]` is a VectorNode immediately after the `self` on the i-th coordinate,
    `self.prev[j]` is a VectorNode immediately before the `self` on the j-th coordinate.
    The `area` is a vector, with its i-th element equal the area of the projection
    of the `coordinate` on the (i-1) subspace.
    The `volume` is the product of the `area` by the difference between the i-th
    coordinate of the self and self.prev[i].
    The `dominated_flag` is used to skip dominated points (see section III.C).
    The VectorNode data structure is introduced in section III.A of the original paper..
    """

    def __init__(
        self, dimension: int, coordinates: tp.Optional[tp.Union[np.ndarray, tp.List[float]]] = None
    ) -> None:
        self.dimension = dimension
        self.coordinates = np.array(coordinates, copy=False)
        self._next: tp.List["VectorNode"] = [self for _ in range(self.dimension)]
        self._prev: tp.List["VectorNode"] = [self for _ in range(self.dimension)]
        self.dominated_flag = 0
        self.area = np.zeros(self.dimension)
        self.volume = np.zeros(self.dimension)

    def __str__(self) -> str:
        return str(self.coordinates)

    def __lt__(self, other: tp.Any) -> bool:
        assert isinstance(other, VectorNode)
        return bool(np.all(self.coordinates < other.coordinates))

    def configure_area(self, dimension: int) -> None:
        self.area[0] = 1.0
        self.area[1 : dimension + 1] = [-self.area[i] * self.coordinates[i] for i in range(dimension)]

    @property
    def next(self) -> tp.List["VectorNode"]:
        return self._next

    @property
    def prev(self) -> tp.List["VectorNode"]:
        return self._prev

    def pop(self, index: int) -> None:
        """Assigns the references of the self predecessor and successor at
        `index` index to each other, removes the links to the `self` node.
        """
        predecessor = self.prev[index]
        successor = self.next[index]
        assert predecessor is not None and successor is not None
        predecessor.next[index] = successor
        successor.prev[index] = predecessor


class VectorLinkedList:
    """ Linked list structure with list of VectorNodes as elements."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.sentinel = VectorNode(dimension)

    @classmethod
    def create_sorted(cls, dimension: int, points: tp.Any) -> "VectorLinkedList":
        """Instantiate a VectorLinkedList of dimension `dimension`. The list is
        populated by nodes::VectorNode created from `points`. The nodes are sorted
        by i-th coordinate attribute in i-th row."""
        linked_list = cls(dimension)
        nodes = [VectorNode(dimension, coordinates=point) for point in points]
        for i in range(dimension):
            sorted_node = cls.sort_by_index(nodes, i)
            linked_list.extend(sorted_node, i)
        return linked_list

    @staticmethod
    def sort_by_index(node_list: tp.List[VectorNode], dimension_index: int) -> tp.List[VectorNode]:
        """Returns a sorted list of `VectorNode`, with the sorting key defined by the
        `dimension_index`-th coordinates of the nodes in the `node_list`."""
        return sorted(node_list, key=lambda node: node.coordinates[dimension_index])

    def __str__(self) -> str:
        string = [str([str(node) for node in self.iterate(dimension)]) for dimension in range(self.dimension)]
        return "\n".join(string)

    def __len__(self) -> int:
        return self.dimension

    def chain_length(self, index: int) -> int:
        length = sum(1 for _ in self.iterate(index))
        return length

    def append(self, node: VectorNode, index: int) -> None:
        """ Append a node to the `index`-th position."""
        current_last = self.sentinel.prev[index]
        assert current_last is not None
        node.next[index] = self.sentinel
        node.prev[index] = current_last
        self.sentinel.prev[index] = node
        current_last.next[index] = node

    def extend(self, nodes: tp.List[VectorNode], index: int) -> None:
        """Extends the VectorLinkedList with a list of nodes
        at `index` position"""
        for node in nodes:
            self.append(node, index)

    @staticmethod
    def update_coordinate_bounds(bounds: np.ndarray, node: VectorNode, index: int) -> np.ndarray:
        for i in range(index):
            if bounds[i] > node.coordinates[i]:
                bounds[i] = node.coordinates[i]
        return bounds

    def pop(self, node: VectorNode, index: int) -> VectorNode:
        """Removes and returns 'node' from all lists at the
        positions from 0 in index (exclusively)."""
        for i in range(index):
            node.pop(i)

        return node

    def reinsert(self, node: VectorNode, index: int) -> None:
        """
        Inserts 'node' at the position it had before it was removed
        in all lists at the positions from 0 in index (exclusively).
        This method assumes that the next and previous nodes of the
        node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node

    def iterate(self, index: int, start: tp.Optional[VectorNode] = None) -> tp.Iterator[VectorNode]:
        if start is None:
            node = self.sentinel.next[index]
        else:
            node = start
        while node is not self.sentinel:
            assert node is not None
            yield node
            node = node.next[index]

    def reverse_iterate(self, index: int, start: tp.Optional[VectorNode] = None) -> tp.Iterator[VectorNode]:
        if start is None:
            node = self.sentinel.prev[index]
        else:
            node = start
        while node is not self.sentinel:
            assert node is not None
            yield node
            node = node.prev[index]


class HypervolumeIndicator:
    """Core class to calculate the hypervolme value of a set of points.
    As introduced in the original paper, "the indicator is a measure of
    the region which is simultaneously dominated by a set of points P,
    and bounded by a reference point r = `self.reference_bounds`. It is
    a union of axis-aligned hyper-rectangles with one common vertex, r."

    To calculate the hypervolume indicator, initialize an instance of the
    HypervolumeIndicator; the hypervolume of a set of points P is calculated
    by HypervolumeIndicator.compute(points = P) method.

    For the algorithm, refer to the section III and Algorithms 2, 3 of the
    paper `An Improved Dimension-Sweep Algorithm for the Hypervolume Indicator`
    by C.M. Fonseca et all, IEEE Congress on Evolutionary Computation, 2006.
    """

    def __init__(self, reference_point: np.ndarray) -> None:
        self.reference_point = np.array(reference_point, copy=False)
        self.dimension = self.reference_point.size
        self.reference_bounds = np.full(self.dimension, -np.inf)
        self._multilist: tp.Optional[VectorLinkedList] = None

    @property
    def multilist(self) -> VectorLinkedList:
        assert self._multilist is not None
        return self._multilist

    def compute(self, points: tp.Union[tp.List[np.ndarray], np.ndarray]) -> float:
        points = points - self.reference_point
        self.reference_bounds = np.full(self.dimension, -np.inf)
        self._multilist = VectorLinkedList.create_sorted(self.dimension, points)
        hypervolume = self.recursive_hypervolume(self.dimension - 1)
        return hypervolume

    def plane_hypervolume(self) -> float:
        """Calculates the hypervolume on a two dimensional plane. The algorithm
        is described in Section III-A of the original paper."""
        dimension = 1
        hypervolume = 0.0
        h = self.multilist.sentinel.next[dimension].coordinates[dimension - 1]
        for node in self.multilist.iterate(dimension):
            next_node = node.next[dimension]
            if next_node is self.multilist.sentinel:
                break
            hypervolume += h * (node.coordinates[dimension] - next_node.coordinates[dimension])
            h = min(h, next_node.coordinates[dimension - 1])
        last_node = self.multilist.sentinel.prev[dimension]
        hypervolume += h * last_node.coordinates[dimension]
        return hypervolume

    def recursive_hypervolume(self, dimension: int) -> float:
        """Recursive hypervolume computation. The algorithm is provided by Algorithm 3.
        of the original paper."""
        if self.multilist.chain_length(dimension - 1) == 0:
            return 0
        assert self.multilist is not None
        if dimension == 0:
            return -float(self.multilist.sentinel.next[0].coordinates[0])

        if dimension == 1:
            return self.plane_hypervolume()

        # Line 4
        for node in self.multilist.reverse_iterate(dimension):
            assert node is not None
            if node.dominated_flag < dimension:
                node.dominated_flag = 0
        # Line 5
        hypervolume = 0.0
        # Line 6
        current_node = self.multilist.sentinel.prev[dimension]
        # Lines 7 to 12
        for node in self.multilist.reverse_iterate(dimension, start=current_node):
            assert node is not None
            current_node = node
            if self.multilist.chain_length(dimension - 1) > 1 and (
                node.coordinates[dimension] > self.reference_bounds[dimension]
                or node.prev[dimension].coordinates[dimension] >= self.reference_bounds[dimension]
            ):
                # Line 9
                self.reference_bounds = self.multilist.update_coordinate_bounds(
                    self.reference_bounds, node, dimension
                )  # type: ignore
                # Line 10
                self.multilist.pop(node, dimension)
                # Line 11
                # front_size -= 1
            else:
                break

        # Line 13
        if self.multilist.chain_length(dimension - 1) > 1:
            # Line 14
            hypervolume = current_node.prev[dimension].volume[dimension]
            hypervolume += current_node.prev[dimension].area[dimension] * (
                current_node.coordinates[dimension] - current_node.prev[dimension].coordinates[dimension]
            )
        else:
            current_node.configure_area(dimension)

        # Line 15
        current_node.volume[dimension] = hypervolume
        # Line 16
        self.skip_dominated_points(current_node, dimension)

        # Line 17
        for node in self.multilist.iterate(dimension, start=current_node.next[dimension]):
            assert node is not None
            # Line 18
            hypervolume += node.prev[dimension].area[dimension] * (
                node.coordinates[dimension] - node.prev[dimension].coordinates[dimension]
            )
            # Line 19
            self.reference_bounds[dimension] = node.coordinates[dimension]
            # Line 20
            self.reference_bounds = self.multilist.update_coordinate_bounds(
                self.reference_bounds, node, dimension
            )  # type: ignore
            # Line 21
            self.multilist.reinsert(node, dimension)
            # Line 22
            # front_size += 1
            # Line 25
            node.volume[dimension] = hypervolume
            # Line 26
            self.skip_dominated_points(node, dimension)

        # Line 27
        last_node = self.multilist.sentinel.prev[dimension]
        hypervolume -= last_node.area[dimension] * last_node.coordinates[dimension]
        return hypervolume

    def skip_dominated_points(self, node: VectorNode, dimension: int) -> None:
        """ Implements Algorithm 2, _skipdom_, for skipping dominated points."""
        if node.dominated_flag >= dimension:
            node.area[dimension] = node.prev[dimension].area[dimension]
        else:
            node.area[dimension] = self.recursive_hypervolume(dimension - 1)
            if node.area[dimension] <= node.prev[dimension].area[dimension]:
                node.dominated_flag = dimension
