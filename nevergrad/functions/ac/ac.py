# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Approximate Rocket Simulation
Based on
https://raw.githubusercontent.com/purdue-orbital/rocket-simulation/master/Simulation2.py
"""

import math
import pyproj
import numpy as np
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
        import sys

        # _=[sys.path.append(i) for i in ['.', '..']]

        path = get_filepath("champion_climate.txt")
        wdf = prepare_weather(path)

        def run_model(smts, max_irr_season, year1, year2):
            """
            funciton to run model and return results for given set of soil moisture targets
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

        import numpy as np  # import numpy library

        def evaluate(smts) -> float:  # ,max_irr_season,test=False):
            """
            funciton to run model and calculate reward (yield) for given set of soil moisture targets
            """
            max_irr_season = self.max_irr_seasonal
            assert len(smts) == self.num_smts
            # run model
            out = run_model(smts, max_irr_season, year1=2016, year2=2018)
            # get yields and total irrigation
            yld = out["Yield (tonne/ha)"].mean()
            tirr = out["Seasonal irrigation (mm)"].mean()

            reward = yld

            # return either the negative reward (for the optimization)
            # or the yield and total irrigation (for analysis)
            # if test:
            #    return yld,tirr,reward
            # else:
            return -reward

        return evaluate(smts)


#        def get_starting_point(num_smts,max_irr_season,num_searches):
#            """
#            find good starting threshold(s) for optimization
#            """
#
#            # get random SMT's
#            x0list = np.random.rand(num_searches,num_smts)*100
#            rlist=[]
#            # evaluate random SMT's
#            for xtest in x0list:
#                r = evaluate(xtest,max_irr_season,)
#                rlist.append(r)
#
#            # save best SMT
#            x0=x0list[np.argmin(rlist)]
#
#            return x0

# from scipy.optimize import fmin

# def optimize(num_smts,max_irr_season,num_searches=100):
#     """
#     optimize thresholds to be profit maximising
#     """
#     # get starting optimization strategy
#     x0=get_starting_point(num_smts,max_irr_season,num_searches)
#     # run optimization
#     res = fmin(evaluate, x0,disp=0,args=(max_irr_season,))
#     # reshape array
#     smts= res.squeeze()
#     # evaluate optimal strategy
#     return smts

# smts=optimize(self.num_smts, self.max_irr_seasonal)

# print(evaluate(smts,max_irr_season,True))

# #from tqdm.notebook import tqdm # progress bar
#
# opt_smts=[]
# yld_list=[]
# tirr_list=[]
# for max_irr in (range(0,500,50)):
#
#
#     # find optimal thresholds and save to list
#     smts=optimize(4,max_irr)
#     opt_smts.append(smts)
#
#     # save the optimal yield and total irrigation
#     yld,tirr,_=evaluate(smts,max_irr,True)
#     yld_list.append(yld)
#     tirr_list.append(tirr)
#
# import matplotlib.pyplot as plt
#
# # create plot
# fig,ax=plt.subplots(1,1,figsize=(13,8))
#
# # plot results
# ax.scatter(tirr_list,yld_list)
# ax.plot(tirr_list,yld_list)
#
# # labels
# ax.set_xlabel('Total Irrigation (ha-mm)',fontsize=18)
# ax.set_ylabel('Yield (tonne/ha)',fontsize=18)
# ax.set_xlim([-20,600])
# ax.set_ylim([2,15.5])
#
# # annotate with optimal thresholds
# bbox = dict(boxstyle="round",fc="1")
# offset = [15,15,15, 15,15,-125,-100,  -5, 10,10]
# yoffset= [0,-5,-10,-15, -15,  0,  10,15, -20,10]
# for i,smt in enumerate(opt_smts):
#     smt=smt.clip(0,100)
#     ax.annotate('(%.0f, %.0f, %.0f, %.0f)'%(smt[0],smt[1],smt[2],smt[3]),
#                 (tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
#                 bbox=bbox,fontsize=12)
#
#
