import numpy as np
from typing import List, Iterator, Optional


class VectorNode:
    def __init__(self, dimension: int, coordinate: List[float] = None) -> None:
        self.dimension = dimension
        self.coordinate = coordinate
        self.next = [None for _ in range(self.dimension)]
        self.prev = [None for _ in range(self.dimension)]
        self.dominated_flag = 0
        self.area = np.zeros(self.dimension)
        self.volume = np.zeros(self.dimension)

    def __str__(self):
        return str(self.coordinate)

    def __lt__(self, other):
        return np.all(self.coordinate < other.coordinate)

    def configure_area(self, dimension: int) -> None:
        self.area[0] = 1.0
        self.area[1 : dimension + 1] = [
            -self.area[i] * self.coordinate[i] for i in range(dimension)
        ]

    def pop(self, index: int) -> None:
        """ Assigns the references of the self predecessor and successor at
        `index` index to each other, removes the links to the `self` node.
        """
        predecessor = self.prev[index]
        successor = self.next[index]
        predecessor.next[index] = successor
        successor.prev[index] = predecessor


class VectorLinkedList:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.sentinel = VectorNode(dimension)

        self.sentinel.prev = [self.sentinel for _ in range(dimension)]
        self.sentinel.next = [self.sentinel for _ in range(dimension)]

    def __str__(self):
        string = [
            str([str(node) for node in self.iterate(dimension)])
            for dimension in range(self.dimension)
        ]
        return "\n".join(string)

    def __len__(self) -> int:
        return self.dimension

    def append(self, node: VectorNode, index: int) -> None:
        """ Append a node to the `index`-th position."""
        current_last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = current_last
        self.sentinel.prev[index] = node
        current_last.next[index] = node

    def extend(self, nodes: List[Optional[VectorNode]], index: int) -> None:
        """ Extends the VectorLinkedList with a list of nodes
        at `index` position"""
        for node in nodes:
            self.append(node, index)

    @staticmethod
    def update_coordinate_bounds(bounds: List, node: VectorNode, index: int) -> List:
        if bounds is None:
            return
        for i in range(index):
            if bounds[i] > node.coordinate[i]:
                bounds[i] = node.coordinate[i]
        return bounds

    def pop(self, node: VectorNode, index: int) -> VectorNode:
        """ Removes and returns 'node' from all lists at the
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

    def iterate(self, index: int, start: VectorNode = None) -> Iterator[Optional[VectorNode]]:
        if start is None:
            node = self.sentinel.next[index]
        else:
            node = start
        while node is not self.sentinel:
            yield node
            node = node.next[index]

    def reverse_iterate(self, index: int, start: VectorNode = None) -> Iterator[Optional[VectorNode]]:
        if start is None:
            node = self.sentinel.prev[index]
        else:
            node = start
        while node is not self.sentinel:
            yield node
            node = node.prev[index]


class HypervolumeIndicator:
    def __init__(self, reference_point) -> None:
        self.reference_point = reference_point
        self.dimension = len(self.reference_point)
        self.reference_bounds = np.full(self.dimension, -np.inf)
        self.multilist = None

    def compute(self, points) -> float:
        points = points - self.reference_point

        self.multilist = self.construct_linkedlist(points)
        hypervolume = self.recursive_hypervolume(
            self.dimension - 1, len(points), self.reference_bounds
        )
        return hypervolume

    def construct_linkedlist(self, points) -> VectorLinkedList:
        linkedlist = VectorLinkedList(self.dimension)
        nodes = [VectorNode(self.dimension, point) for point in points]
        for i in range(self.dimension):
            sorted_node = self.sort_by_index(nodes, i)
            linkedlist.extend(sorted_node, i)

        return linkedlist

    def sort_by_index(self, node_list: List[Optional[VectorNode]], dimension_index: int) -> List[Optional[VectorNode]]:
        """ Returns a sorted list of `VectorNode`, with the sorting key defined by the
        `dimension_index`-th coordinates of the nodes in the `node_list`."""
        return sorted(node_list, key=lambda node: node.coordinate[dimension_index])

    def plane_hypervolume(self, dimension: int) -> float:
        """ Calculates the hypervolume on a two dimensional plane. The algorithm
        is described in Section III-A of the original paper. """
        assert dimension == 1
        hypervolume = 0.0
        h = self.multilist.sentinel.next[dimension].coordinate[dimension - 1]
        for node in self.multilist.iterate(dimension):
            next_node = node.next[dimension]
            if next_node is self.multilist.sentinel:
                break
            hypervolume += h * (node.coordinate[dimension] - next_node.coordinate[dimension])
            h = min(h, next_node.coordinate[dimension - 1])
        hypervolume += h * node.coordinate[dimension]
        return hypervolume

    def recursive_hypervolume(self, dimension: int, front_size: int, bounds: List[float]):
        """ Recursive hypervolume computation. The algorithm is provided by Algorithm 3.
        of the original paper."""
        if front_size == 0:
            return 0

        if dimension == 0:
            return -self.multilist.sentinel.next[0].coordinate[0]

        if dimension == 1:
            return self.plane_hypervolume(dimension)

        # Line 4
        for node in self.multilist.reverse_iterate(dimension):
            if node.dominated_flag < dimension:
                node.dominated_flag = 0
        # Line 5
        hypervolume = 0.0
        # Lines 6 to 12
        for node in self.multilist.reverse_iterate(dimension):
            if front_size > 1 and (
                node.coordinate[dimension] > bounds[dimension]
                or node.prev[dimension].coordinate[dimension] >= bounds[dimension]
            ):
                # Line 9
                bounds = self.multilist.update_coordinate_bounds(bounds, node, dimension)
                # Line 10
                self.multilist.pop(node, dimension)
                # Line 11
                front_size -= 1
            else:
                break

        # Line 13
        if front_size > 1:
            # Line 14
            hypervolume = node.prev[dimension].volume[dimension]
            hypervolume += node.prev[dimension].area[dimension] * (
                node.coordinate[dimension] - node.prev[dimension].coordinate[dimension]
            )
        else:
            node.configure_area(dimension)

        # Line 15
        node.volume[dimension] = hypervolume
        # Line 16
        self.skip_dominated_points(node, dimension, front_size, bounds)

        # Line 17
        for node in self.multilist.iterate(dimension, start=node.next[dimension]):
            # Line 18
            hypervolume += node.prev[dimension].area[dimension] * (
                node.coordinate[dimension] - node.prev[dimension].coordinate[dimension]
            )
            # Line 19
            bounds[dimension] = node.coordinate[dimension]
            # Line 20
            bounds = self.multilist.update_coordinate_bounds(bounds, node, dimension)
            # Line 21
            self.multilist.reinsert(node, dimension)
            # Line 22
            front_size += 1
            # Line 25
            node.volume[dimension] = hypervolume
            # Line 26
            self.skip_dominated_points(node, dimension, front_size, bounds)

        # Line 27
        hypervolume -= node.area[dimension] * node.coordinate[dimension]
        return hypervolume

    def skip_dominated_points(
        self, node: VectorNode, dimension: int, front_size: int, bounds: List[float]
    ):
        """ Implements Algorithm 2, _skipdom_, for skipping dominated points."""
        if node.dominated_flag >= dimension:
            node.area[dimension] = node.prev[dimension].area[dimension]
        else:
            node.area[dimension] = self.recursive_hypervolume(dimension - 1, front_size, bounds)
            if node.area[dimension] <= node.prev[dimension].area[dimension]:
                node.dominated_flag = dimension


if __name__ == "__main__":
    reference = np.array([79, 89, 99])
    hv = HypervolumeIndicator(reference)
    front = [
        (110, 110, 100),  # -0 + distance
        (110, 90, 87),  # -0 + distance
        (80, 80, 36),  # -400 + distance
        (50, 50, 55),
        (105, 30, 43),
    ]
    volume = hv.compute(front)

    from nevergrad.functions.multiobjective.pyhv import _HyperVolume
    reference_volume = _HyperVolume(reference).compute(front)
    assert volume == reference_volume
