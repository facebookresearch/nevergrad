# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Approximate crop Simulation
Based on
https://raw.githubusercontent.com/purdue-orbital/pcse-simulation/master/Simulation2.py
"""


from pathlib import Path
import urllib.request  # Necessary for people who will uncomment the part using data under EUPL license.
import numpy as np
import nevergrad as ng
from ..base import ArrayExperimentFunction
import os
import pandas as pd
import yaml
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import WOFOST72SiteDataProvider
from pcse.base import ParameterProvider


# pylint: disable=too-many-locals,too-many-statements


class Irrigation(ArrayExperimentFunction):
    def __init__(self, symmetry: int) -> None:
        data_dir = Path(__file__).with_name("data")
        #urllib.request.urlretrieve(
        #    "https://raw.githubusercontent.com/ajwdewit/pcse_notebooks/master/data/soil/ec3.soil",
        #    str(data_dir) + "/soil/ec3.soil",
        #)
        self.soil = CABOFileReader(os.path.join(data_dir, "soil", "ec3.soil"))
        param = ng.p.Array(shape=(8,), lower=(0.0), upper=(1.0)).set_name("irrigation8")
        super().__init__(self.leaf_area_index, parametrization=param, symmetry=symmetry)
        known_longitudes = {'Saint-Leger-Bridereix': 1.5887348, 'Dun-Le-Palestel': 1.6641173, 'Kolkata':
        88.35769124388872, 'Antananarivo': 47.5255809, 'Santiago': -70.6504502, 'Lome': 1.215829, 'Cairo': 31.2357257,
        'Ouagadougou': -1.5270944, 'Yamoussoukro': -5.273263, 'Yaounde': 11.5213344, 'Kiev': 30.5241361}
        known_latitudes = {'Saint-Leger-Bridereix': 46.2861759, 'Dun-Le-Palestel': 46.3052049, 'Kolkata': 22.5414185,
        'Antananarivo': -18.9100122, 'Santiago': -33.4377756, 'Lome': 6.130419, 'Cairo': 30.0443879, 'Ouagadougou':
        12.3681873, 'Yamoussoukro': 6.809107, 'Yaounde': 3.8689867, 'Kiev': 50.4500336}
        for k in range(1000):
            self.address = np.random.RandomState(symmetry+3*k).choice(
                [
                    "Saint-Leger-Bridereix",
                    "Dun-Le-Palestel",
                    "Kolkata",
                    "Antananarivo",
                    "Santiago",
                    "Lome",
                    "Cairo",
                    "Ouagadougou",
                    "Yamoussoukro",
                    "Yaounde",
                    "Kiev",
                ]
            )
            if self.address in known_latitudes and self.address in known_longitudes:
                self.weatherdataprovider = NASAPowerWeatherDataProvider(latitude=known_latitudes[self.address], longitude=known_longitudes[self.address])
            else:           
                assert False
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="NG/PCSE")
                self.location = geolocator.geocode(self.address)
                self.weatherdataprovider = NASAPowerWeatherDataProvider(
                    latitude=self.location.latitude, longitude=self.location.longitude
                )

            cropd = YAMLCropDataProvider()
            crop_types = [c for c in cropd.crop_types if "obacco" not in c]
            self.cropname = np.random.RandomState(symmetry+3*k+1).choice(crop_types)
            self.cropvariety = np.random.RandomState(symmetry+3*k+2).choice(list(cropd.get_crops_varieties()[self.cropname])
            )
            # We check if the problem is challenging.
            print(f"testing {symmetry}: {k} {self.address} {self.cropvariety}")
            crop = YAMLCropDataProvider()
            if os.environ.get("CIRCLECI", False):
                raise ng.errors.UnsupportedExperiment("No HTTP request in CircleCI")
            site = WOFOST72SiteDataProvider(WAV=100, CO2=360)
            self.parameterprovider = ParameterProvider(soildata=self.soil, cropdata=crop, sitedata=site)
            self.crop = YAMLCropDataProvider()
            v = [self.leaf_area_index(np.random.rand(8)) for _ in range(5)]
            if min(v) != max(v):
                break
        print(f"we work on {self.cropname} with variety {self.cropvariety} in {self.address}.")

    def leaf_area_index(self, x: np.ndarray):
        d0 = int(1.01 + 29.98 * x[0])
        d1 = int(1.01 + 30.98 * x[1])
        d2 = int(1.01 + 30.98 * x[2])
        d3 = int(1.01 + 29.98 * x[3])
        a0 = 15.0 * x[4] / (x[4] + x[5] + x[6] + x[7])
        a1 = 15.0 * x[5] / (x[4] + x[5] + x[6] + x[7])
        a2 = 15.0 * x[6] / (x[4] + x[5] + x[6] + x[7])
        a3 = 15.0 * x[7] / (x[4] + x[5] + x[6] + x[7])





        yaml_agro = f"""
        - 2006-01-01:
            CropCalendar:
                crop_name: {self.cropname}
                variety_name: {self.cropvariety}
                crop_start_date: 2006-03-31
                crop_start_type: emergence
                crop_end_date: 2006-10-20
                crop_end_type: harvest
                max_duration: 300
            TimedEvents:
            -   event_signal: irrigate
                name: Irrigation application table
                comment: All irrigation amounts in cm
                events_table:
                - 2006-06-{d0:02}: {{amount: {a0}, efficiency: 0.7}}
                - 2006-07-{d1:02}: {{amount: {a1}, efficiency: 0.7}}
                - 2006-08-{d2:02}: {{amount: {a2}, efficiency: 0.7}}
                - 2006-09-{d3:02}: {{amount: {a3}, efficiency: 0.7}}
            StateEvents: null
        """
        try:
            agromanagement = yaml.safe_load(yaml_agro)
            wofost = Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider, agromanagement)
            wofost.run_till_terminate()
        except Exception as e:
            return float("inf")
            #assert (
            #    False
            #), f"Problem!\n Dates: {d0} {d1} {d2} {d3},\n amounts: {a0}, {a1}, {a2}, {a3}\n  ({e}).\n"
            #raise e

        output = wofost.get_output()
        df = pd.DataFrame(output).set_index("day")
        df.tail()

        return -sum([float(o["LAI"]) for o in output if o["LAI"] is not None])
