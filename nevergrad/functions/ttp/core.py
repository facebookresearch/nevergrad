# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import nevergrad as ng
from nevergrad.parametrization import parameter as p
from .. import base


class TTPSolution:
    def __init__(self, tspTour, packingPlan):

        self.tspTour = tspTour
        self.packingPlan = packingPlan

        self.fp = float("-inf")
        self.ft = float("inf")
        self.ftraw = math.inf
        self.ob = float("-inf")
        self.wend = float("inf")
        self.wendUsed = float("inf")
        self.computationTime = math.inf

    def reset(self):

        self.fp = float("-inf")
        self.ft = float("inf")
        self.ftraw = math.inf
        self.ob = float("-inf")

        self.wend = float("inf")
        self.wendUsed = float("inf")
        self.computationTime = math.inf


class TTPInstance(base.ExperimentFunction):
    def __init__(self, file: str = "./instances/a20_n95_01.ttp") -> None:

        self.filepath = file
        reader = open(self.filepath, "r")

        lines = reader.readlines()
        count = 0
        while count < len(lines):
            line = lines[count]
            if line.startswith("PROBLEM NAME"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.problemName = line

            if line.startswith("KNAPSACK DATA TYPE"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.knapsackDataType = line

            if line.startswith("DIMENSION"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.numberOfNodes = int(line)

            if line.startswith("NUMBER OF ITEMS"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.numberOfItems = int(line)

            if line.startswith("CAPACITY OF KNAPSACK"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.capacityOfKnapsack = int(line)

            if line.startswith("MIN SPEED"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.minSpeed = float(line)

            if line.startswith("MAX SPEED"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.maxSpeed = float(line)

            if line.startswith("RENTING RATIO"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.rentingRatio = float(line)

            if line.startswith("EDGE_WEIGHT_TYPE"):
                line = line[line.index(":") + 1 :]
                line = line.strip()
                self.edgeWeightType = line

            if line.startswith("NODE_COORD_SECTION"):
                self.nodes = []

                for i in range(0, self.numberOfNodes):
                    count += 1 + i - i
                    line = lines[count]
                    splittedLine = line.split()
                    col = []

                    for j in range(0, len(splittedLine)):
                        temp = float(splittedLine[j])
                        if j == 0:
                            temp = temp - 1
                        col.append(temp)
                    self.nodes.append(col)

            if line.startswith("ITEMS SECTION"):
                self.items = []
                for i in range(0, self.numberOfItems):
                    count += 1
                    line = lines[count]
                    splittedLine = line.split()
                    col = []

                    for j in range(0, len(splittedLine)):
                        temp = int(splittedLine[j])
                        # adjust city number by 1
                        if j == 0:
                            temp = temp - 1
                        if j == 3:
                            temp = temp - 1
                        col.append(temp)
                    self.items.append(col)

            count += 1
        reader.close()

        init1 = np.zeros(self.numberOfNodes - 1)
        p1 = p.Array(init=init1)
        p1.set_bounds(0, 4)
        p1.set_name("tspTour")

        init2 = 0 * np.ones(self.numberOfItems)
        p2 = p.Array(init=init2)
        p2.set_integer_casting()
        p2.set_bounds(0, 1)
        p2.set_name("packingPlan")

        instru = ng.p.Instrumentation(tspTour=p1, packingPlan=p2).set_name("")

        super().__init__(self.evaluate, instru)

    def evaluate(self, tspTour: np.ndarray, packingPlan: np.ndarray) -> float:

        np.reshape(tspTour, (1, -1))
        tspTour = np.argsort(tspTour)

        tspTour = tspTour + 1
        tspTL = tspTour.tolist()
        tspTL.insert(0, 0)
        tspTL.append(0)

        packingPlan = packingPlan.tolist()

        solution = TTPSolution(tspTL, packingPlan)

        tour = solution.tspTour
        z = solution.packingPlan
        weightofKnapsack = self.capacityOfKnapsack
        rentRate = self.rentingRatio
        vmin = self.minSpeed
        vmax = self.maxSpeed
        solution.ftraw = 0

        if tour[0] != tour[-1]:
            print("ERROR: The last city must be the same as the first city")
            solution.reset()
            return float(-1)

        wc = float(0)
        solution.ft = 0
        solution.fp = 0

        itemsPerCity = int(len(solution.packingPlan) / (len(solution.tspTour) - 2))

        for i in range(0, len(tour) - 1):
            currentCityTEMP = tour[i]
            currentCity = currentCityTEMP - 1

            if i > 0:
                for itemNumber in range(0, itemsPerCity):
                    indexOfPackingPlan = (i - 1) * itemsPerCity + itemNumber

                    itemIndex = currentCity + itemNumber * (self.numberOfNodes - 1)

                    if z[indexOfPackingPlan] == 1:
                        currentWC = self.items[itemIndex][2]
                        wc = wc + currentWC

                        currentFP = self.items[itemIndex][1]
                        solution.fp = solution.fp + currentFP

            h = (i + 1) % (len(tour) - 1)

            distance = int(math.ceil(self.distances(tour[i], tour[h])))

            solution.ftraw += distance

            if wc > self.capacityOfKnapsack:
                solution.ft = solution.ft + (distance / vmin)
            else:
                solution.ft = solution.ft + (distance / (vmax - (wc * (vmax - vmin) / weightofKnapsack)))

        solution.wendUsed = wc
        solution.wend = weightofKnapsack - wc
        solution.ob = -(solution.fp - (solution.ft * rentRate))

        if wc > self.capacityOfKnapsack:
            solution.ob = solution.ob + 1000 * (wc - self.capacityOfKnapsack)

        return float(solution.ob)

    def distances(self, i, j) -> float:

        result = float(0)
        result = math.sqrt(
            (self.nodes[i][1] - self.nodes[j][1]) * (self.nodes[i][1] - self.nodes[j][1])
            + (self.nodes[i][2] - self.nodes[j][2]) * (self.nodes[i][2] - self.nodes[j][2])
        )

        return float(result)
