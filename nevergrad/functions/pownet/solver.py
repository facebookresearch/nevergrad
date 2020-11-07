import os
import importlib.util
import requests
from pyomo.opt import SolverFactory
from pyomo.core import Var
from operator import itemgetter
from datetime import datetime
import nevergrad as ng
from nevergrad.functions.pyomo import core

import git
import pypownetr.model

def download_dataset(dataset_name="cambodian", force_update=False):
    #git clone <repository> <path>
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pypownet')
    if not os.path.exists(dataset_path):
        git.Repo.clone_from("https://github.com/pacowong/pypownet.git", dataset_path, branch="main")
    elif force_update:
        git.cmd.Git(dataset_path).pull('origin', 'main') #Check update
    if dataset_name == "cambodian":
        return os.path.join(dataset_path, 'pypownetr', 'datasets', 'kamal0013', 'camb_2016')
    else:
        raise NotImplementedError

def _load_sim_data(dst, src, nodes, horizon_hours, day):
    for z in nodes:
        for i in range(1, horizon_hours+1):
            dst[z,i] = src[z,(day-1)*24+i]


def solve_pownet_subproblems(pyomo_model, model_data_path, solver, year=2016, start_day=1, last_day=365):
    """
    simulation year, start(1-365) and end(1-365) days of simulation
    """
    
    instance = pyomo_model.create_instance(model_data_path)

    ###solver and number of threads to use for simulation
    H = instance.HorizonHours
    K = range(1, H+1)

    ###Run simulation and save outputs
    #Containers to store results
    on = []
    switch = []

    mwh = []
    hydro = []
    solar = []
    wind = []

    hydro_import = []

    srsv = []
    nrsv = []

    vlt_angle = []

    system_cost = []

    for day in range(start_day, last_day+1):
        if hasattr(instance, 'd_nodes'):
            for z in instance.d_nodes:
                # Load Demand and Reserve time series data
                for i in K:
                    instance.HorizonDemand[z, i] = instance.SimDemand[z, (day-1)*24+i]
                    instance.HorizonReserves[i] = instance.SimReserves[(day-1)*24+i] 
                
        if hasattr(instance, 'h_nodes'):
            for z in instance.h_nodes:
                # Load Hydropower time series data
                for i in K:
                    instance.HorizonHydro[z, i] = instance.SimHydro[z, (day-1)*24+i]
                
        if hasattr(instance, 's_nodes'):
            for z in instance.s_nodes:
                # Load Solar time series data
                for i in K:
                    instance.HorizonSolar[z, i] = instance.SimSolar[z, (day-1)*24+i]
                
        if hasattr(instance, 'w_nodes'):
            # _load_sim_data(instance.HorizonWind, instance.SimWind, instance.w_nodes, H, day)
            for z in instance.w_nodes:
                # Load Wind time series data
                for i in K:
                    instance.HorizonWind[z, i] = instance.SimWind[z, (day-1)*24+i]
                
        if hasattr(instance, 'h_imports'):
            
            for z in instance.h_imports:
                # Load Hydropower time series data
                for i in K:
                    instance.HorizonHydroImport[z,i] = instance.SimHydroImport[z,(day-1)*24+i]     
        
        func = core.Pyomo(instance)
        optimizer = ng.optimizers.CMA(parametrization=func.parametrization, budget=10)
        recommendation = optimizer.minimize(func.function)
        print(recommendation) #Low system_cost is better
        #result = solver.solve(instance) ##,tee=True to check number of variables
        # instance.display()
        # instance.solutions.load_from(result)
        # system_cost.append((day, instance.SystemCost.value()))
    
        # #The following section is for storing and sorting results
        # for v in instance.component_objects(Var, active=True):
        #     varobject = getattr(instance, str(v))
        #     a = str(v)
        #     if a=='hydro':      
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 if index[0] in instance.h_nodes:
        #                     hydro.append((index[0],index[1]+((day-1)*24),varobject[index].value))

        #     elif a=='solar':
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 if index[0] in instance.s_nodes:
        #                     solar.append((index[0],index[1]+((day-1)*24),varobject[index].value))   

        #     elif a=='wind':
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 if index[0] in instance.w_nodes:
        #                     wind.append((index[0],index[1]+((day-1)*24),varobject[index].value))   

        #     elif a=='hydro_import':      
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 if index[0] in instance.h_imports:
        #                     hydro_import.append((index[0],index[1]+((day-1)*24),varobject[index].value))   

        #     elif a=='vlt_angle':
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 if index[0] in instance.nodes:
        #                     vlt_angle.append((index[0],index[1]+((day-1)*24),varobject[index].value))   

        #     elif a=='mwh':  
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 mwh.append((index[0],index[1]+((day-1)*24),varobject[index].value))                            

        #     elif a=='on':       
        #         ini_on_ = {}  
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 on.append((index[0],index[1]+((day-1)*24),varobject[index].value))
        #             if int(index[1])==24:
        #                 ini_on_[index[0]] = varobject[index].value    

        #     elif a=='switch':  
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 switch.append((index[0],index[1]+((day-1)*24),varobject[index].value))

        #     elif a=='srsv':    
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 srsv.append((index[0],index[1]+((day-1)*24),varobject[index].value))
                            
        #     elif a=='nrsv':   
        #         for index in varobject:
        #             if int(index[1]>0 and index[1]<25):
        #                 nrsv.append((index[0],index[1]+((day-1)*24),varobject[index].value))                             
        
        # Update initialization values for "on" 
        # for z in instance.Generators:
        #     instance.ini_on[z] = round(ini_on_[z]) #for next day

        
        print(day)
        print(str(datetime.now()))

    return {
        'on': on,
        'switch': switch, 
        'mwh': mwh,
        'hydro': hydro, 
        'solar': solar, 
        'wind': wind, 
        'hydro_import': hydro_import, 
        'srsv': srsv, 
        'nrsv': nrsv, 
        'vlt_angle': vlt_angle,
        'system_cost': system_cost
    }

def pownet_solver(model_type, year=2016, start_day=1, end_day=2):
    # year: simulation year, start and end of simulation
    # start_day: first day of simulation (1 to 365)
    # end_day: last day of simulation (1 to 365)
    assert start_day>=1 and start_day<=365
    assert end_day>start_day and end_day<=365
    
    if model_type == "cambodian":    
        cambodian_dataset_dir = download_dataset(dataset_name="cambodian", force_update=False)
        pownet_pyomo = pypownetr.model.PowerNetPyomoModelCambodian(dataset_dir=cambodian_dataset_dir, year=year)
        pyomo_model = pownet_pyomo.create_model()
        #model_data_path = pownet_pyomo.get_data_path()
        #print(f'Save the transformed model data to {model_data_path}')
        model_data_path = 'C:\\Doc\\repos\\nevergrad\\nevergrad\\functions\\pownet\\pypownet\\pypownetr\\datasets\\kamal0013\\camb_2016\\temp.dat'
        print(f'Save the transformed model data to {model_data_path}')
        answers = solve_pownet_subproblems(pyomo_model, model_data_path, solver=None, year=year, start_day=start_day, last_day=end_day)


        #from .pownet_cambodian.Model_withdata.pownet_model import model as pownet_model_cambodian #Cannot load twice!!! TODO: use importlib
        #instance = pownet_model_cambodian.create_instance(pownet_dat)
    else:
        raise NotImplementedError

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


def solve_pownet_problem():
    #cambodian_dataset_dir = download_dataset(dataset_name="cambodian", force_update=False)
    year = 2016
    start = 1
    end = 2
    #pownet_pyomo = pypownetr.model.PowerNetPyomoModelCambodian(dataset_dir=cambodian_dataset_dir, year=year)
    pownet_solver(model_type="cambodian", year=year, start_day=start, end_day=end)


if __name__ == "__main__":
    solve_pownet_problem()
