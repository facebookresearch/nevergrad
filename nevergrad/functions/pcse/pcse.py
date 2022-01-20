# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Approximate Pcse Simulation
Based on
https://raw.githubusercontent.com/purdue-orbital/pcse-simulation/master/Simulation2.py
"""


import math
import pyproj
import numpy as np
from nevergrad.parametrization import parameter
from ..base import ArrayExperimentFunction

# pylint: disable=too-many-locals,too-many-statements


class Pcse(ArrayExperimentFunction):


    def __init__(self, symmetry: int = 0) -> None:
        import sys
        import matplotlib
        import matplotlib.pyplot as plt
        import yaml
        import pandas as pd
        import numpy as np
        from itertools import product
        #from progressbar import printProgressBar
        
        import pcse
        from pcse.models import Wofost72_PP
        from pcse.base import ParameterProvider
        from pcse.db import NASAPowerWeatherDataProvider
        from pcse.fileinput import YAMLAgroManagementReader, YAMLCropDataProvider
        from pcse.util import WOFOST72SiteDataProvider, DummySoilDataProvider
    
        # Weather data for Netherlands
        wdp = NASAPowerWeatherDataProvider(latitude=52, longitude=5)
        # Standard crop parameter library
        cropd = YAMLCropDataProvider()
        # We don't need soil for potential production, so we use dummy values
        soild = DummySoilDataProvider()
        # Some site parameters
        sited = WOFOST72SiteDataProvider(WAV=50, CO2=360.)
        # Package everyting into a single parameter object
        params = ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild)
        
        # Here we define the agromanagement for sugar beet
        agro_yaml = """
        - 2006-01-01:
            CropCalendar:
                crop_name: sugarbeet
                variety_name: Sugarbeet_603
                crop_start_date: 2006-03-31
                crop_start_type: emergence
                crop_end_date: 2006-10-20
                crop_end_type: harvest
                max_duration: 300
            TimedEvents: null
            StateEvents: null
        """
        agro = yaml.load(agro_yaml)
        
        wofost = Wofost72_PP(params, wdp, agro)
        wofost.run_till_terminate()
        df = pd.DataFrame(wofost.get_output())
        df.index = pd.to_datetime(df.day)
        df.tail()
        
        # get daily observations for those
        ix = (df.index.dayofweek == 0) & (df.LAI.notnull())
        df_pseudo_obs = df.loc[ix]
        fig, axes = plt.subplots(figsize=(12,8))
        axes.plot_date(df_pseudo_obs.index, df_pseudo_obs.LAI)
        r = axes.set_title("Pseudo LAI observations")
        
        class ModelRerunner(object):
            """Reruns a given model with different values of parameters TWDI and SPAN.
            
            Returns a pandas DataFrame with simulation results of the model with given
            parameter values.
            """
            parameters = ["TDWI", "SPAN"]
            
            def __init__(self, params, wdp, agro):
                self.params = params
                self.wdp = wdp
                self.agro = agro
                
            def __call__(self, par_values):
                # Check if correct number of parameter values were provided
                if len(par_values) != len(self.parameters):
                    msg = "Optimizing %i parameters, but only % values were provided!" % (len(self.parameters, len(par_values)))
                    raise RuntimeError(msg)
                # Clear any existing overrides
                self.params.clear_override()
                # Set overrides for the new parameter values
                for parname, value in zip(self.parameters, par_values):
                    self.params.set_override(parname, value)
                # Run the model with given parameter values
                wofost = Wofost72_PP(self.params, self.wdp, self.agro)
                wofost.run_till_terminate()
                df = pd.DataFrame(wofost.get_output())
                df.index = pd.to_datetime(df.day)
                return df
        
        class ObjectiveFunctionCalculator(object):
            """Computes the objective function.
            
            This class runs the simulation model with given parameter values and returns the objective
            function as the sum of squared difference between observed and simulated LAI.
        .   """
            
            def __init__(self, params, wdp, agro, observations):
                self.modelrerunner = ModelRerunner(params, wdp, agro)
                self.df_observations = observations
                self.n_calls = 0
               
            def __call__(self, par_values, grad=None):
                """Runs the model and computes the objective function for given par_values.
                
                The input parameter 'grad' must be defined in the function call, but is only
                required for optimization methods where analytical gradients can be computed.
                """
                self.n_calls += 1
                #print(".", end="")
                # Run the model and collect output
                self.df_simulations = self.modelrerunner(par_values)
                # compute the differences by subtracting the DataFrames
                # Note that the dataframes automatically join on the index (dates) and column names
                df_differences = self.df_simulations - self.df_observations
                # Compute the RMSE on the LAI column
                obj_func = np.sqrt(np.mean(df_differences.LAI**2))
                return obj_func
        
        objfunc_calculator = ObjectiveFunctionCalculator(params, wdp, agro, df_pseudo_obs)
        # defaults = [cropd["TDWI"], cropd["SPAN"]]
        # error = objfunc_calculator(defaults)
        # print("Objective function value with default parameters (%s): %s" % (defaults, error))
    
        param = ng.p.Array(shape=(2,), lower=(TDWI_range[0], SPAN_range[0]), upper=(TDWI_range[1], SPAN_range[1]))
        super().__init__(objfunc_calculator, parametrization=param, symmetry=symmetry)
