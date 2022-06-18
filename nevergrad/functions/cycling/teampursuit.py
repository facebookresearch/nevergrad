# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import math


class teampursuit(ABC):

    # constants
    friction_coefficient = 0.0025
    drafting_coefficients = [0.75, 0.65, 0.55]
    gravitational_acceleration = 9.80665
    time_step = 0.1

    transition_time = 0.12

    relative_humidity = 0.5

    temperature = 20.0
    barometric_pressure = 1013.25

    air_density = None
    team = None

    def __init__(self):
        self.update_air_density()

    def set_temperature(self, temperature):
        if temperature < 0.0 or temperature > 40:
            raise ValueError("Temperature must be in range 0-40C")
        else:
            self.temperature = temperature
            self.update_air_density()

    def set_barometric_pressure(self, barometric_pressure):
        if barometric_pressure < 800.0 or barometric_pressure > 1200.0:
            raise ValueError("Barometric pressure must be in the range 800-1200 hPa")
        else:
            self.barometric_pressure = barometric_pressure
            self.update_air_density()

    def set_relative_humidity(self, relative_humidity):
        if relative_humidity < 0.0 or relative_humidity > 1.0:
            raise ValueError("Relative humidity must be in the range 0-1")
        else:
            self.relative_humidity = relative_humidity
            self.update_air_density()

    def set_height(self, cyclist_id, height):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            self.team[cyclist_id].set_height(height)

    def set_weight(self, cyclist_id, weight):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            self.team[cyclist_id].set_weight(weight)

    def set_mean_maximum_power(self, cyclist_id, mean_maximum_power):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            self.team[cyclist_id].set_mean_maximum_power(mean_maximum_power)

    def get_temperature(self):
        return self.temperature

    def get_barometric_pressure(self):
        return self.barometric_pressure

    def get_relative_humidity(self):
        return self.relative_humidity

    def get_height(self, cyclist_id):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            return self.team[cyclist_id].get_height()

    def get_weight(self, cyclist_id):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            return self.team[cyclist_id].get_weight()

    def get_mean_maximum_power(self, cyclist_id):
        if cyclist_id >= len(self.team):
            raise ValueError(f"Cyclist identifier must be in the range 0-{len(self.team)}")
        else:
            return self.team[cyclist_id].get_mean_maximum_power()

    @abstractmethod
    def simulate(self, transition_strategy, pacing_strategy):
        pass

    def update_air_density(self):
        pp_water_vapour = (
            100
            * self.relative_humidity
            * (
                6.1078
                * math.pow(
                    10,
                    (((7.5 * (self.temperature + 273.15)) - 2048.625)) / (self.temperature + 273.15 - 35.85),
                )
            )
        )
        pp_dry_air = 100 * self.barometric_pressure - pp_water_vapour
        self.air_density = (pp_dry_air / (287.058 * (self.temperature + 273.15))) + (
            pp_water_vapour / (461.495 * (self.temperature + 273.15))
        )

    def cyclists_remaining(self):
        cyclists_remaining = 0
        for i in range(0, len(self.team)):
            if self.team[i].get_remaining_energy() > 0.0:
                cyclists_remaining += 1
            else:
                self.team[i].set_position(0)
        return cyclists_remaining

    def leader(self):
        for i in range(0, len(self.team)):
            if (self.team[i].get_position()) == 1:
                return self.team[i]
        return None

    def transition(self):
        for i in range(0, len(self.team)):
            if self.team[i].get_position() > 0:
                if self.team[i].get_position() == 1:
                    self.team[i].set_position(self.cyclists_remaining())
                else:
                    self.team[i].set_position(self.team[i].get_position() - 1)
