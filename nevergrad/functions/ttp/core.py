# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import numpy as np
import nevergrad as ng
from pathlib import Path
from nevergrad.parametrization import parameter as p
from .. import base


class TTPSolution:
    def __init__(self, tsp_tour, packing_plan):

        self.tsp_tour = tsp_tour
        self.packing_plan = packing_plan

        self.fp = float("-inf")
        self.ft = float("inf")
        self.ft_raw = math.inf
        self.ob = float("-inf")
        self.wend = float("inf")
        self.wend_used = float("inf")
        self.computation_time = math.inf

    def reset(self):

        self.fp = float("-inf")
        self.ft = float("inf")
        self.ft_raw = math.inf
        self.ob = float("-inf")

        self.wend = float("inf")
        self.wend_used = float("inf")
        self.computation_time = math.inf


class TTPInstance(base.ExperimentFunction):
    def __init__(self, file: str = "a20_n95_01.ttp") -> None:

        self.file_path = Path(os.path.dirname(__file__) + "/instances/" + file)

        with self.file_path.open() as reader:

            lines = reader.readlines()
            count = 0
            while count < len(lines):
                line = lines[count]
                if line.startswith("PROBLEM NAME"):
                    line = line.split(":")[1].strip()
                    self.problem_name = line

                if line.startswith("KNAPSACK DATA TYPE"):
                    line = line.split(":")[1].strip()
                    self.knapsack_data_type = line

                if line.startswith("DIMENSION"):
                    line = line.split(":")[1].strip()
                    self.number_of_nodes = int(line)

                if line.startswith("NUMBER OF ITEMS"):
                    line = line.split(":")[1].strip()
                    self.number_of_items = int(line)

                if line.startswith("CAPACITY OF KNAPSACK"):
                    line = line.split(":")[1].strip()
                    self.capacity_of_knapsack = int(line)

                if line.startswith("MIN SPEED"):
                    line = line.split(":")[1].strip()
                    self.min_speed = float(line)

                if line.startswith("MAX SPEED"):
                    line = line.split(":")[1].strip()
                    self.max_speed = float(line)

                if line.startswith("RENTING RATIO"):
                    line = line.split(":")[1].strip()
                    self.renting_ratio = float(line)

                if line.startswith("EDGE_WEIGHT_TYPE"):
                    line = line.split(":")[1].strip()
                    self.edge_weight_type = line

                if line.startswith("NODE_COORD_SECTION"):
                    self.nodes = []

                    for i in range(0, self.number_of_nodes):
                        count += 1 + i - i
                        line = lines[count]
                        splitted_line = line.split()
                        col = []

                        for j in range(0, len(splitted_line)):
                            temp = float(splitted_line[j])
                            if j == 0:
                                temp = temp - 1
                            col.append(temp)
                        self.nodes.append(col)

                if line.startswith("ITEMS SECTION"):
                    self.items = []
                    for i in range(0, self.number_of_items):
                        count += 1
                        line = lines[count]
                        splitted_line = line.split()
                        col = []

                        for j in range(0, len(splitted_line)):
                            temp = int(splitted_line[j])
                            # adjust city number by 1
                            if j == 0:
                                temp = temp - 1
                            if j == 3:
                                temp = temp - 1
                            col.append(temp)
                        self.items.append(col)

                count += 1

        init1 = np.zeros(self.number_of_nodes - 1)
        p1 = p.Array(init=init1, lower=0, upper=4)

        # p2 = p.Choice([0,1], repetitions = self.number_of_items)
        init2 = np.zeros(self.number_of_items)
        p2 = p.Array(init=init2, lower=0, upper=1).set_integer_casting()

        instru = ng.p.Instrumentation(tsp_tour=p1, packing_plan=p2).set_name("")

        super().__init__(self.evaluate, instru)

    def evaluate(self, tsp_tour: np.ndarray, packing_plan: np.ndarray) -> float:

        np.reshape(tsp_tour, (1, -1))
        tsp_tour = np.argsort(tsp_tour)

        tsp_tour = tsp_tour + 1
        tsp_tl = tsp_tour.tolist()
        tsp_tl.insert(0, 0)
        tsp_tl.append(0)

        packing_plan = packing_plan.tolist()

        solution = TTPSolution(tsp_tl, packing_plan)

        tour = solution.tsp_tour
        z = solution.packing_plan
        weight_of_knapsack = self.capacity_of_knapsack
        rentRate = self.renting_ratio
        v_min = self.min_speed
        v_max = self.max_speed
        solution.ft_raw = 0

        if tour[0] != tour[-1]:
            print("ERROR: The last city must be the same as the first city")
            solution.reset()
            return float(-1)

        wc = float(0)
        solution.ft = 0
        solution.fp = 0

        items_per_city = int(len(solution.packing_plan) / (len(solution.tsp_tour) - 2))

        for i in range(0, len(tour) - 1):
            current_city_temp = tour[i]
            current_city = current_city_temp - 1

            if i > 0:
                for item_number in range(0, items_per_city):
                    index_of_packing_plan = (i - 1) * items_per_city + item_number

                    item_index = current_city + item_number * (self.number_of_nodes - 1)

                    if z[index_of_packing_plan] == 1:
                        current_wc = self.items[item_index][2]
                        wc = wc + current_wc

                        current_fp = self.items[item_index][1]
                        solution.fp = solution.fp + current_fp

            h = (i + 1) % (len(tour) - 1)

            distance = int(math.ceil(self.distances(tour[i], tour[h])))

            solution.ft_raw += distance

            if wc > self.capacity_of_knapsack:
                solution.ft = solution.ft + (distance / v_min)
            else:
                solution.ft = solution.ft + (distance / (v_max - (wc * (v_max - v_min) / weight_of_knapsack)))

        solution.wend_used = wc
        solution.wend = weight_of_knapsack - wc
        solution.ob = -(solution.fp - (solution.ft * rentRate))

        if wc > self.capacity_of_knapsack:
            solution.ob = solution.ob + 1000 * (wc - self.capacity_of_knapsack)

        return float(solution.ob)

    def distances(self, i, j) -> float:

        result = float(0)
        result = math.sqrt(
            (self.nodes[i][1] - self.nodes[j][1]) * (self.nodes[i][1] - self.nodes[j][1])
            + (self.nodes[i][2] - self.nodes[j][2]) * (self.nodes[i][2] - self.nodes[j][2])
        )

        return float(result)
