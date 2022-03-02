# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Approximate crop Simulation
Based on
https://raw.githubusercontent.com/purdue-orbital/pcse-simulation/master/Simulation2.py
"""


from pathlib import Path
import urllib.request
import numpy as np
import warnings
import nevergrad as ng
from ..base import ArrayExperimentFunction

# pylint: disable=too-many-locals,too-many-statements


class Irrigation(ArrayExperimentFunction):
    def __init__(self, symmetry: int) -> None:
        import nevergrad as ng

        param = ng.p.Array(shape=(8,), lower=(0.0), upper=(1.0)).set_name("irrigation8")
        super().__init__(leaf_area_index, parametrization=param, symmetry=symmetry)


import numpy as np


def leaf_area_index(x: np.ndarray):
    d0 = int(1.01 + 29.98 * x[0])
    d1 = int(1.01 + 30.98 * x[1])
    d2 = int(1.01 + 30.98 * x[2])
    d3 = int(1.01 + 29.98 * x[3])
    a0 = 15.0 * x[4] / (x[4] + x[5] + x[6] + x[7])
    a1 = 15.0 * x[5] / (x[4] + x[5] + x[6] + x[7])
    a2 = 15.0 * x[6] / (x[4] + x[5] + x[6] + x[7])
    a3 = 15.0 * x[7] / (x[4] + x[5] + x[6] + x[7])
    import os, sys

    # import matplotlib
    # matplotlib.style.use("ggplot")
    # import matplotlib.pyplot as plt
    import pandas as pd
    import yaml

    import pcse
    from pcse.models import Wofost72_WLP_FD
    from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

    # from pcse.db import NASAPowerWeatherDataProvider
    from pcse.util import WOFOST72SiteDataProvider
    from pcse.base import ParameterProvider

    # data_dir = os.path.join(os.getcwd(), "data")
    data_dir = Path(__file__).with_name("data")

    print("This notebook was built with:")
    print("python version: %s " % sys.version)
    print("PCSE version: %s" % pcse.__version__)

    crop = YAMLCropDataProvider()
    if os.environ.get("CIRCLECI", False):
        raise ng.errors.UnsupportedExperiment("No HTTP request in CircleCI")
    warnings.warn("Check that you have no problem with the EUPL license.")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ajwdewit/ggcmi/master/pcse/doc/ec3.soil",
        str(data_dir) + "/soil/ec3.soil",
    )
    soil = CABOFileReader(os.path.join(data_dir, "soil", "ec3.soil"))
    site = WOFOST72SiteDataProvider(WAV=100, CO2=360)
    parameterprovider = ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)

    crop = YAMLCropDataProvider()

    from pcse.fileinput import ExcelWeatherDataProvider

    warnings.warn("Check that you have no problem with the EUPL license.")
    urllib.request.urlretrieve(
        "https://pcse.readthedocs.io/en/stable/_downloads/78c1c853e9911098db9e3d8e6f362550/nl1.xlsx",
        str(data_dir) + "/meteo/nl1.xlsx",
    )
    weatherfile = os.path.join(data_dir, "meteo", "nl1.xlsx")
    weatherdataprovider = ExcelWeatherDataProvider(weatherfile)

    yaml_agro = f"""
    - 2006-01-01:
        CropCalendar:
            crop_name: sugarbeet
            variety_name: Sugarbeet_603
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
        wofost = Wofost72_WLP_FD(parameterprovider, weatherdataprovider, agromanagement)
        wofost.run_till_terminate()
    except Exception as e:
        assert False, f"Problem!\n Dates: {d0} {d1} {d2} {d3},\n amounts: {a0}, {a1}, {a2}, {a3}\n  ({e}).\n"
        raise e

    output = wofost.get_output()
    df = pd.DataFrame(output).set_index("day")
    df.tail()

    # print(output)
    # print(len(output))
    return -sum([float(o["LAI"]) for o in output if o["LAI"] is not None])
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    # df['LAI'].plot(ax=axes[0], title="Leaf Area Index")
    # df['SM'].plot(ax=axes[1], title="Root zone soil moisture")
    # fig.autofmt_xdate()
