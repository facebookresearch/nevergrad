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
import time
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


# dollars per kilogram
crop_value_dollars_per_kg = {}
crop_value_dollars_per_kg["barley"]= 9.6  # Kenya
crop_value_dollars_per_kg["cassava"]= 2    # Kenya
crop_value_dollars_per_kg["chick_pea"]= 0.395   # Kenya
crop_value_dollars_per_kg["cotton"]= 1.83  # Kenya
crop_value_dollars_per_kg["cow_pea"]= 0.185  # Kenya
crop_value_dollars_per_kg["faba_bean"]= 0.1825  # no idea where
crop_value_dollars_per_kg["ground_nut"]= 0.66  # Kenya
crop_value_dollars_per_kg["maize"]= 0.37  # Kenya
crop_value_dollars_per_kg["millet"]= 0.485  # Kenya
crop_value_dollars_per_kg["mung_bean"]= 0.33  # no idea where, unclear#
crop_value_dollars_per_kg["pigeon_pea"]= 0.679  # not clear, kenya
crop_value_dollars_per_kg["potato"]= 0.55  # kenya, difficult
crop_value_dollars_per_kg["rape_seed"]=7.1   # kenya (waow ?)
crop_value_dollars_per_kg["rice"]=0.325  # kenya
crop_value_dollars_per_kg["sorghum"]= 0.25  # kenya
crop_value_dollars_per_kg["soy_bean"]=(0.68+0.93)/2.  # kenya
crop_value_dollars_per_kg["sugar_beet"]=(0.75+0.69)/2.  # kenya
crop_value_dollars_per_kg["sugar_cane"]=0.34   # kenya, roughly
crop_value_dollars_per_kg["sunflower"]= 1.69  # kenya
crop_value_dollars_per_kg["sweet_potato"]= 1.15  # kenya
crop_value_dollars_per_kg["wheat"]=(0.19+0.42)/2.  # roughly, kenya


def lev(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def get_crop_cost(cropname: str):
    min_lev = 100000000
    if cropname in crop_value_dollars_per_kg:
        return crop_value_dollars_per_kg[cropname]
    for k in crop_value_dollars_per_kg:
        dist = lev(cropname, k)
        if dist < min_lev:
            min_lev = dist
            best_crop = k
    print(f"best crop for {cropname} is {best_crop}")
    crop_value_dollars_per_kg[cropname] = crop_value_dollars_per_kg[best_crop]
    return crop_value_dollars_per_kg[cropname]

WPD = {}
CURRENT_BEST = {}
CURRENT_BEST_ARGUMENT = {}


class Irrigation(ArrayExperimentFunction):
    variant_choice = {}
    def __init__(self, symmetry:int, benin: bool, variety_choice: bool, rice: bool, multi_crop: bool, address=None, year_min: int = 2006, year_max: int = 2006) -> None:
        self.rice = rice
        if year_max < year_min and year_max == 2006:
            year_max = year_min
        self.year_min = year_min
        self.year_max = year_max
        data_dir = Path(__file__).with_name("data")
        try:
            self.soil = CABOFileReader(os.path.join(data_dir, "soil", "ec3.soil"))
        except:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/ajwdewit/pcse_notebooks/master/data/soil/ec3.soil",
                str(data_dir) + "/soil/ec3.soil",
            )
            self.soil = CABOFileReader(os.path.join(data_dir, "soil", "ec3.soil"))
        self.this_dimension = 8
        if rice or (variety_choice and not multi_crop):
            self.this_dimension = 9
        if multi_crop:
            self.this_dimension = 10
            assert not rice

        param = ng.p.Array(shape=(self.this_dimension,), lower=(0.0), upper=(0.99999999)).set_name("irrigation8")
        super().__init__(self.meta_total_yield, parametrization=param, symmetry=symmetry)
        if os.environ.get("CIRCLECI", False):
            raise ng.errors.UnsupportedExperiment("No HTTP request in CircleCI")
#                    "Cotonou",
#                    "Lokossa",
#                    "Allada",
#                    "Abomey",
#                    "Pobe",
#                    "Aplahoue",
#                    "Dassa-Zoume",
#                    "Parakou",
#                    "Djougou",
#                    "Kandi",
#                    "Natitingou",
        known_longitudes = {'Saint-Leger-Bridereix': 1.5887348, 'Dun-Le-Palestel': 1.6641173, 'Kolkata':
        88.35769124388872, 'Antananarivo': 47.5255809, 'Santiago': -70.6504502, 'Lome': 1.215829, 'Cairo': 31.2357257,
        'Ouagadougou': -1.5270944, 'Yamoussoukro': -5.273263, 'Yaounde': 11.5213344, 'Kiev': 30.5241361, 'Porto-Novo':
        2.6289}
        known_latitudes = {'Saint-Leger-Bridereix': 46.2861759, 'Dun-Le-Palestel': 46.3052049, 'Kolkata': 22.5414185,
        'Antananarivo': -18.9100122, 'Santiago': -33.4377756, 'Lome': 6.130419, 'Cairo': 30.0443879, 'Ouagadougou':
        12.3681873, 'Yamoussoukro': 6.809107, 'Yaounde': 3.8689867, 'Kiev': 50.4500336, 'Porto-Novo': 6.4969}
        known_longitudes['Cotonou'] = 2.4252507
        known_latitudes['Cotonou'] = 6.3676953
        known_longitudes['Lokossa'] = 1.7171404
        known_latitudes['Lokossa'] = 6.6458524
        known_longitudes['Allada'] = 2.1511876
        known_latitudes['Allada'] = 6.6658411
        known_longitudes['Abomey'] = 1.9828803672675925
        known_latitudes['Abomey'] = 7.165446
        known_longitudes['Pobe'] = -1.751602
        known_latitudes['Pobe'] = 13.882217
        known_longitudes['Aplahoue'] = 1.7041012
        known_latitudes['Aplahoue'] = 6.9489244
        known_longitudes['Dassa-Zoume'] = 2.183606
        known_latitudes['Dassa-Zoume'] = 7.7815402
        known_longitudes['Parakou'] = 2.6278258
        known_latitudes['Parakou'] = 9.3400159
        known_longitudes['Djougou'] = 1.6651614
        known_latitudes['Djougou'] = 9.7106683
        known_longitudes['Kandi'] = 88.11640162351831
        known_latitudes['Kandi'] = 24.00952125
        known_longitudes['Natitingou'] = 1.383540986380074
        known_latitudes['Natitingou'] = 10.251408300000001
        self.cropd = YAMLCropDataProvider()
        for k in range(1000):
            if symmetry in self.variant_choice and k < self.variant_choice[symmetry]:
                continue
            self.address = np.random.RandomState(symmetry+3*k).choice(
                [
                    "Saint-Leger-Bridereix",
                    "Dun-Le-Palestel",
                    "Porto-Novo",
                    "Kolkata",
                    "Antananarivo",
                    "Santiago",
                    "Lome",
                    "Cairo",
                    "Ouagadougou",
                    "Yamoussoukro",
                    "Yaounde",
                    "Porto-Novo",
                    "Kiev",
                ] if not benin else [
                    "Porto-Novo",
                    "Cotonou",
                    "Lokossa",
                    "Allada",
                    "Abomey",
                    "Pobe",
                    "Aplahoue",
                    "Dassa-Zoume",
                    "Parakou",
                    "Djougou",
                    "Kandi",
                    "Natitingou",
                ]
            )
            if address is not None:
                 self.address = str(address)
                 if len(address) == 2:
                     known_latitudes[self.address] = address[0]
                     known_longitudes[self.address] = address[1]
                 
            if self.address in known_latitudes and self.address in known_longitudes and self.address not in WPD:
                for k in range(10):
                    try:
                        WPD[self.address] = NASAPowerWeatherDataProvider(latitude=known_latitudes[self.address], longitude=known_longitudes[self.address])
                        break
                    except:
                        time.sleep(10 * (2 **k))
            if self.address in WPD:
                self.weatherdataprovider = WPD[self.address]
            else:           
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="NG/PCSE")
                self.location = geolocator.geocode(self.address)
                self.weatherdataprovider = NASAPowerWeatherDataProvider(
                    latitude=self.location.latitude, longitude=self.location.longitude
                )
                WPD[self.address] = self.weatherdataprovider
            self.set_data(symmetry, k, rice)
            v = [self.meta_total_yield(np.random.rand(self.this_dimension)) for _ in range(5)]
            if min(v) != max(v):
                break
            self.variant_choice[symmetry] = k
        print(f"we work on {self.cropname} with variety {self.cropvariety} in {self.address}.")


    def set_data(self, symmetry: int, k: int, rice: bool):
        crop_types = [crop for crop, variety in self.cropd.get_crops_varieties().items()]
        crop_types = [c for c in crop_types if "obacco" not in c]
        if rice:
            crop_types = ["rice"]
        self.crop_types = crop_types
        self.cropname = np.random.RandomState(symmetry+3*k+1).choice(crop_types)
        self.total_irrigation = np.random.RandomState(symmetry+3*k+3).choice([15.0, 1.50, 0.15, 150.]) if not rice else 0.15
        self.cropvariety = np.random.RandomState(symmetry+3*k+2).choice(list(self.cropd.get_crops_varieties()[self.cropname])
        )
        # We check if the problem is challenging.
        #print(f"testing {symmetry}: {k} {self.address} {self.cropvariety}")
        site = WOFOST72SiteDataProvider(WAV=100, CO2=360)
        self.parameterprovider = ParameterProvider(soildata=self.soil, cropdata=self.cropd, sitedata=site)


    def meta_total_yield(self, x: np.ndarray):
        lai = 0.
        for year in range(self.year_min, self.year_max+1):
            print(f"working on {year}")
            lai += self.total_yield(x, year)
        specifier = self.address + "_" + str(self.total_irrigation)
        if not self.this_dimension == 10:
            specifier += "_" + str(self.cropname)
        if self.dimension == 8:
            specifier += "_" + self.cropvariety
        if specifier not in CURRENT_BEST:
            CURRENT_BEST[specifier] = 0.
        if lai > CURRENT_BEST[specifier]:
            CURRENT_BEST[specifier] = lai
            argument = str(x)
            if self.dimension > 9:
                argument += "_" + self.cropname
            if self.dimension > 8:
                argument += "_" + self.cropvariety
            CURRENT_BEST_ARGUMENT[specifier] = argument
            print(f"for <{specifier}> we recommend {CURRENT_BEST_ARGUMENT[specifier]} and get {lai}")

        return -lai

    def total_yield(self, x: np.ndarray, year:int=2006):
        d0 = int(1.01 + 29.98 * x[0])
        d1 = int(1.01 + 30.98 * x[1])
        d2 = int(1.01 + 30.98 * x[2])
        d3 = int(1.01 + 29.98 * x[3])
        c = self.total_irrigation
        if len(x) == 10:
            c = 0
        a0 = c * x[4] / (x[4] + x[5] + x[6] + x[7])
        a1 = c * x[5] / (x[4] + x[5] + x[6] + x[7])
        a2 = c * x[6] / (x[4] + x[5] + x[6] + x[7])
        a3 = c * x[7] / (x[4] + x[5] + x[6] + x[7])
        if len(x) > 8:
            if self.this_dimension == 10:
                assert len(x) == 10
                self.cropname = self.crop_types[int(x[9] * len(self.crop_types))]
            else:
                assert len(x) == 9, f"my x has size {len(x)}, it is {x}"
            varieties = list(self.cropd.get_crops_varieties()[self.cropname])
            self.cropvariety = varieties[int(x[8] * len(varieties))]

        yaml_agro = f"""
        - {year}-01-01:
            CropCalendar:
                crop_name: {self.cropname}
                variety_name: {self.cropvariety}
                crop_start_date: {year}-03-31
                crop_start_type: emergence
                crop_end_date: {year}-10-20
                crop_end_type: harvest
                max_duration: 300
            TimedEvents:
            -   event_signal: irrigate
                name: Irrigation application table
                comment: All irrigation amounts in cm
                events_table:
                - {year}-06-{d0:02}: {{amount: {a0}, efficiency: 0.7}}
                - {year}-07-{d1:02}: {{amount: {a1}, efficiency: 0.7}}
                - {year}-08-{d2:02}: {{amount: {a2}, efficiency: 0.7}}
                - {year}-09-{d3:02}: {{amount: {a3}, efficiency: 0.7}}
            StateEvents: null
        """
        print(list(self.weatherdataprovider.keys()))
        assert False
        try:
            agromanagement = yaml.safe_load(yaml_agro)
            wofost = Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider, agromanagement)
            wofost.run_till_terminate()
#            for i in range(10):
#                try:
#                    wofost = Wofost72_WLP_FD(self.parameterprovider, self.weatherdataprovider, agromanagement)
#                    wofost.run_till_terminate()
#                    break
#                except:
#                    print(f"failure {i} for {yaml_agro}")
#                    time.sleep(2 ** i)
#                return -float("inf")
        except Exception as e:
            print(e)
            return -float("inf")
            #assert (
            #    False
            #), f"Problem!\n Dates: {d0} {d1} {d2} {d3},\n amounts: {a0}, {a1}, {a2}, {a3}\n  ({e}).\n"
            #raise e

        output = wofost.get_output()
        df = pd.DataFrame(output).set_index("day")
        df.tail()

        lai = sum([float(o["TWSO"]) for o in output if o["TWSO"] is not None])
        value = lai * get_crop_cost(self.cropname)
        return value
