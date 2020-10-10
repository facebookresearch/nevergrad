import os
import importlib.util
import requests
from pyomo.opt import SolverFactory
from pyomo.core import Var
from operator import itemgetter
from datetime import datetime
import nevergrad as ng
from nevergrad.functions.pyomo import core

def download_dataset(dataset_name):
    if dataset_name=='cambodian':
        dataset_path = os.path.dirname(os.path.realpath(__file__))
        zipfile_path = os.path.join(dataset_path, "cambodian.zip")
        if not os.path.exists(zipfile_path):
            myfile = requests.get("https://github.com/kamal0013/PowNet/archive/v1.2.zip")
            open(zipfile_path, 'wb').write(myfile.content)
        if not os.path.isdir(os.path.join(dataset_path, "pownet_cambodian")):
            import zipfile
            zip_ref = zipfile.ZipFile(zipfile_path, 'r')
            zip_ref.extractall(dataset_path)
            zip_ref.close()
            os.rename("PowNet-1.2", "pownet_cambodian")
        return
    raise NotImplementedError


def _load_sim_data(dst, src, nodes, horizon_hours, day):
    for z in nodes:
        for i in range(1, horizon_hours+1):
            dst[z,i] = src[z,(day-1)*24+i]


def pownet_solver(pownet_dat, model_type, year=2016, start_day=1, end_day=2):
    # year: simulation year, start and end of simulation
    # start_day: first day of simulation (1 to 365)
    # end_day: last day of simulation (1 to 365)
    assert start_day>=1 and start_day<=365
    assert end_day>start_day and end_day<=365
    
    if model_type == "cambodian":
        from .pownet_cambodian.Model_withdata.pownet_model import model as pownet_model_cambodian #Cannot load twice!!! TODO: use importlib
        instance = pownet_model_cambodian.create_instance(pownet_dat)
    elif model_type == "tiny":
        from .pownet_tiny.model_data.pownet_model import model as pownet_tiny
        instance = pownet_tiny.create_instance(pownet_dat)

    # Run simulation and save outputs
    # Storage
    #on, switch, mwh, hydro, solar, wind, hydro_import, srsv, nrsv, vlt_angle = [[] for _ in range(10)] 
    
    H = instance.HorizonHours

    # Load time series data
    for day in range(start_day, end_day+1):
        # Load Demand and Reserve
        _load_sim_data(instance.HorizonDemand, instance.SimDemand, instance.d_nodes, H, day)
        for i in range(1, H+1):
            instance.HorizonReserves[i] = instance.SimReserves[(day-1)*24+i] 

        # Load power sources
        if hasattr(instance, "SimHydro"):
            _load_sim_data(instance.HorizonHydro, instance.SimHydro, instance.h_nodes, H, day)
        if hasattr(instance, "SimSolar"):
            _load_sim_data(instance.HorizonSolar, instance.SimSolar, instance.s_nodes, H, day)
        if hasattr(instance, "SimWind"):
            _load_sim_data(instance.HorizonWind, instance.SimWind, instance.w_nodes, H, day)
        if hasattr(instance, "SimHydroImport"):
            _load_sim_data(instance.HorizonHydroImport, instance.SimHydroImport, instance.h_imports, H, day)

        func = core.Pyomo(instance)
        optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=10)
        recommendation = optimizer.minimize(func.function)
        print(recommendation)

