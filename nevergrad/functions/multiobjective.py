#    This file is part of DEAP.
#
#    Copyright (C) 2010 Simon Wessing
#    TU Dortmund University
#
#    In personal communication, the original authors authorized DEAP team
#    to use this file under the Lesser General Public License.
#
#    You can find the original library here :
#    http://ls11-www.cs.uni-dortmund.de/_media/rudolph/hypervolume/hv_python.zip
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
#
#    Modified by O. Teytaud as needed for Nevergrad.

from ..instrumentation import InstrumentedFunction
from math import log, floor
import random
from typing import (Tuple, Any, Callable, List, Optional, Dict, ValuesView, Iterator,
                    TypeVar, Generic, Deque, Iterable)
import warnings
import numpy as np
#from ..optimization import base
from ..optimization.base import Candidate
from ..common.typetools import ArrayLike
from ..instrumentation import InstrumentedFunction


ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]


class MultiobjectiveFunction:
    """Given several functions, and threshold on their values (above which solutions are pointless),
    this function returns a single-objective function, correctly instrumented, the minimization of which
    yields a solution to the original multiobjective problem.

    multi_objective_function: objective functions, to be minimized, of the original multiobjective problem.
    upper_bounds: upper_bounds[i] is a threshold above which x is pointless if functions[i](x) > upper_bounds[i].

    Returns an objective function to be minimized (it is a single objective function).
    Warning: this function is not stationary.
    The minimum value obtained for this objective function is -h,
    where h is the hypervolume of the Pareto front obtained, given upper_bounds as a reference point.
    """


    def __init__(self, multiobjective_function: Callable[..., Tuple[float, ...]], upper_bounds: Tuple[float, ...]) -> None:
        self.multiobjective_function = multiobjective_function
        self._hypervolume: Any = _HyperVolume(np.array(upper_bounds))  # type: ignore
        self._points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        self._best_volume = -float("Inf")

    def compute_aggregate_loss(self, losses: Tuple[float, ...], *args: Any, **kwargs: Any) -> float:
        # We compute the hypervolume
        arr_losses = np.array(losses)
        new_volume: float = self._hypervolume.compute([y for _, y in self._points] + [arr_losses])
        if new_volume > self._best_volume:  # This point is good! Let us give him a great mono-fitness value.
            self.best_hypervolume = new_volume
            #if tuple(x) in self.pointset:  # TODO: comparison is not quite possible, is it necessary?
            #    assert v == self.pointset[tuple(x)]  # We work noise-free...
            self._points.append(((args, kwargs), arr_losses))
            #self.pointset[tuple(x)] = v
            return -new_volume
        else:
            # Now we compute for each axis
            distance_to_pareto = float("Inf")
            for _, stored_losses in self._points:
                if (stored_losses <= losses).all():
                    distance_to_pareto = min(distance_to_pareto, min(stored_losses - arr_losses))
            return -new_volume + distance_to_pareto

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        # This part is stationary.
        losses = self.multiobjective_function(*args, **kwargs)
        # The following is not. It should be call locally.
        return self.compute_aggregate_loss(losses, *args, **kwargs)

    @property
    def pareto_front(self) -> List[Tuple[ArgsKwargs, np.ndarray]]:
        new_points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        for argskwargs, losses in self._points:
            #Jshould_be_added = True
            for other_losses in list(self.pointset.values()):
                print(v)
                if (other_losses <= losses).all() and (other_losses < losses).any():
                    #should_be_added = False
                    break
            #if should_be_added:
            #    print(p, ",", val, "should be added...")
            #    new_pointset[p] = val
            #else:
            #    print(p, ",", val, "should not be added...")
        self._points = new_points
        return self._points


class _HyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []


    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        #######
        # fmder: Here it is assumed that every point dominates the reference point
        # for point in front:
        #     # only consider points that dominate the reference point
        #     if weaklyDominates(point, referencePoint):
        #         relevantPoints.append(point)
        relevantPoints = front
        # fmder
        #######
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation

            #######
            # fmder: Assume relevantPoints are numpy array
            # for j in xrange(len(relevantPoints)):
            #     relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in xrange(dimensions)]
            relevantPoints -= referencePoint
            # fmder
            #######

        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume


    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo != None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol


    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = _MultiList(dimensions)
        nodes = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList


    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]



class _MultiList:
    """A special data structure needed by FonsecaHyperVolume.

    It consists of several doubly linked lists that share common nodes. So,
    every node has multiple predecessors and successors, one in every list.

    """

    class Node:

        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next  = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

        def __str__(self):
            return str(self.cargo)

        def __lt__(self, other):
            return all(self.cargo < other.cargo)

    def __init__(self, numberLists):
        """Constructor.

        Builds 'numberLists' doubly linked lists.

        """
        self.numberLists = numberLists
        self.sentinel = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists


    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr


    def __len__(self):
        """Returns the number of lists that are included in this _MultiList."""
        return self.numberLists


    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length


    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node


    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node


    def remove(self, node, index, bounds):
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node


    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
