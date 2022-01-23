# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimization of the FAO crop management model.
Based on 
https://colab.research.google.com/github/thomasdkelly/aquacrop/blob/master/tutorials/AquaCrop_OSPy_Notebook_3.ipynb#scrollTo=YDm931IGNxCb
"""

from nevergrad.parametrization import parameter
from ..base import ExperimentFunction

# pylint: disable=too-many-locals,too-many-statements

# Inspired by
# https://colab.research.google.com/github/thomasdkelly/aquacrop/blob/master/tutorials/AquaCrop_OSPy_Notebook_3.ipynb#scrollTo=YDm931IGNxCb

from aquacrop.classes import *
from aquacrop.core import *


class Ac(ExperimentFunction):
    
    def __init__(self, num_smts: int, max_irr_seasonal: float) -> None:
        self.num_smts = num_smts
        self.max_irr_seasonal = max_irr_seasonal
        super().__init__(self.loss, parametrization=parameter.Array(shape=(num_smts,)))

        
    def loss(self, smts):

        path = get_filepath("champion_climate.txt")
        wdf = prepare_weather(path)

        def run_model(smts, max_irr_season, year1, year2):
            """
            Function to run model and return results for given set of soil moisture targets.
            """

            maize = CropClass("Maize", PlantingDate="05/01")  # define crop
            loam = SoilClass("ClayLoam")  # define soil
            init_wc = InitWCClass(wc_type="Pct", value=[70])  # define initial soil water conditions

            irrmngt = IrrMngtClass(
                IrrMethod=1, SMT=smts, MaxIrrSeason=max_irr_season
            )  # define irrigation management

            # create and run model
            model = AquaCropModel(
                f"{year1}/05/01", f"{year2}/10/31", wdf, loam, maize, IrrMngt=irrmngt, InitWC=init_wc
            )
            model.initialize()
            model.step(till_termination=True)
            return model.Outputs.Final

        
        def evaluate(smts) -> float:  # ,max_irr_season,test=False):
            """
            Function to run model and calculate reward (yield) for given set of soil moisture targets
            """
            max_irr_season = self.max_irr_seasonal
            assert len(smts) == self.num_smts
            out = run_model(smts, max_irr_season, year1=2016, year2=2018)
            # get yields.
            reward = out["Yield (tonne/ha)"].mean()
            return -reward

        return evaluate(smts)
