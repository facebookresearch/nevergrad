# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .teampursuit import teampursuit
from .cyclist import Cyclist
from .simulationresult import simulationresult
import math


class mensteampursuit(teampursuit):

    team_size = 4
    race_distance = 4000
    lap_distance = 250
    race_segments = int((2 * (race_distance / lap_distance)) - 1)
    maximum_transitions = race_segments - 1

    def __init__(self):
        super().__init__()
        self.team = []
        for i in range(0, self.team_size):
            self.team.append(Cyclist(1.75, 75.0, 6.0, self, i + 1, "male"))

    def simulate(self, transition_strategy, pacing_strategy):

        if len(transition_strategy) != self.maximum_transitions:
            raise ValueError(
                f"Transition strategy for the mens team pursuit must have exactly {self.maximum_transitions} elements"
            )
        if len(pacing_strategy) != self.race_segments:
            raise ValueError(
                f"Pacing strategy for the mens team pursuit must have exactly {self.race_segments} elements"
            )
        for i in range(0, self.race_segments):
            if pacing_strategy[i] > Cyclist.max_power or pacing_strategy[i] < Cyclist.min_power:
                raise ValueError(
                    f"All power elements of the pacing strategy must be in the range {Cyclist.min_power}-{Cyclist.max_power} Watts, was {pacing_strategy[i]}"
                )

        for i in range(0, len(self.team)):
            self.team[i].reset()

        velocity_profile = [None] * self.race_segments
        proportion_completed = 0
        race_time = 0

        for i in range(0, self.race_segments):
            if (i == 0) or (i == (self.race_segments - 1)):
                distance = 187.5
            else:
                distance = 125.0
            if super().cyclists_remaining() >= 3:

                if super().cyclists_remaining() == 3:
                    self.validate_order()

                if i >= 1 and transition_strategy[i - 1]:
                    super().transition()
                    race_time += teampursuit.transition_time

                leader = super().leader()
                time = 0.0
                distance_ridden = 0.0
                while distance_ridden < distance:
                    dist = leader.set_pace(pacing_strategy[i])

                    for j in range(0, len(self.team)):
                        if self.team[j].get_position() > 1:
                            self.team[j].follow(dist)

                    if distance_ridden + dist <= distance:
                        distance_ridden += dist
                    else:
                        distance_ridden = distance

                    time += self.time_step

                leader.increase_fatigue()
                for j in range(0, len(self.team)):
                    if self.team[j].get_position() > 1:
                        self.team[j].recover()

                if super().cyclists_remaining() >= 3:
                    velocity_profile[i] = distance / time
                    race_time += time
                    proportion_completed += distance / self.race_distance
                else:
                    race_time = math.inf
            else:
                race_time = math.inf
                break

        remaining_energies = []
        for i in range(0, len(self.team)):
            remaining_energies.append(self.team[i].get_remaining_energy())

        return simulationresult(race_time, proportion_completed, remaining_energies, velocity_profile)

    def validate_order(self):
        for i in range(0, len(self.team)):
            if self.team[i].get_position() == 4:
                self.team[i].set_position(3)
                if self.team[(i + 1) % 4].get_position() == 0:
                    self.team[(i + 2) % 4].set_position(1)
                    self.team[(i + 3) % 4].set_position(2)
                elif self.team[(i + 2) % 4].get_position() == 0:
                    self.team[(i + 3) % 4].set_position(2)
