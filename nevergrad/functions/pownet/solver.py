import os
import urllib
import zipfile
from functools import partial
import urllib.request
import pypownetr.model
import nevergrad as ng
from nevergrad.functions.pyomo import core


def download_dataset(dataset_name="cambodian", force_update=False):
    root_dataset_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "pypownet", "pypownetr", "datasets"
    )
    dataset_paths = {
        "cambodian": {
            "path": os.path.join(root_dataset_path, "kamal0013", "camb_2016"),
            "filename": "kamal0013.zip",
        },
        "artificial": {
            "path": os.path.join(root_dataset_path, "artificial", "camb_2016"),
            "file": "artificial.zip",
        },
    }
    assert dataset_name in dataset_paths
    if not os.path.exists(dataset_paths[dataset_name]["path"]) or force_update:
        os.makedirs(dataset_paths[dataset_name]["path"], exist_ok=True)
        datasetZipPath = os.path.join(root_dataset_path, dataset_paths[dataset_name]["file"])
        urllib.request.urlretrieve(
            f"https://github.com/pacowong/pypownet/raw/main/pypownetr/datasets/{dataset_paths[dataset_name]['file']}",
            datasetZipPath,
        )
        with zipfile.ZipFile(datasetZipPath, "r") as zip_ref:
            zip_ref.extractall(root_dataset_path)
    return dataset_paths[dataset_name]["path"]


def create_pyomo_instance(pownet_pyomo, day=1):
    pyomo_model = pownet_pyomo.create_model()
    model_data_path = pownet_pyomo.get_data_path()
    instance = pyomo_model.create_instance(pyomo_model, model_data_path)
    return configure_instance_by_day(instance, day=day)


def configure_instance_by_day(instance, day=1):
    H = instance.HorizonHours
    K = range(1, H + 1)
    if hasattr(instance, "d_nodes"):
        for z in instance.d_nodes:
            # Load Demand and Reserve time series data
            for i in K:
                instance.HorizonDemand[z, i] = instance.SimDemand[z, (day - 1) * 24 + i]
                instance.HorizonReserves[i] = instance.SimReserves[(day - 1) * 24 + i]
    if hasattr(instance, "h_nodes"):
        for z in instance.h_nodes:
            # Load Hydropower time series data
            for i in K:
                instance.HorizonHydro[z, i] = instance.SimHydro[z, (day - 1) * 24 + i]
    if hasattr(instance, "s_nodes"):
        for z in instance.s_nodes:
            # Load Solar time series data
            for i in K:
                instance.HorizonSolar[z, i] = instance.SimSolar[z, (day - 1) * 24 + i]
    if hasattr(instance, "w_nodes"):
        for z in instance.w_nodes:
            # Load Wind time series data
            for i in K:
                instance.HorizonWind[z, i] = instance.SimWind[z, (day - 1) * 24 + i]
    if hasattr(instance, "h_imports"):
        for z in instance.h_imports:
            # Load Hydropower time series data
            for i in K:
                instance.HorizonHydroImport[z, i] = instance.SimHydroImport[z, (day - 1) * 24 + i]
    return instance


def solve_pownet_subproblems(pyomo_model, model_data_path, solver_func, start_day=1, last_day=365):
    """
    simulation year, start(1-365) and end(1-365) days of simulation
    """
    instance = pyomo_model.create_instance(model_data_path)
    ###Run simulation and save outputs
    # Containers to store results
    recommendations = []
    for day in range(start_day, last_day + 1):
        func = core.Pyomo(configure_instance_by_day(instance, day=day))
        optimizer = solver_func(parametrization=func.parametrization)
        recommendations.append(optimizer.minimize(func.function))
    return solve_pownet_subproblems


def get_pownet_model(location: str, year=2016):
    assert location in ["cambodian", "artificial"]
    dataset_dir = download_dataset(dataset_name=location, force_update=True)
    if location == "cambodian":
        pownet_pyomo = pypownetr.model.PowerNetPyomoModelCambodian(dataset_dir=dataset_dir, year=year)
        return pownet_pyomo
    elif location == "artificial":
        pownet_pyomo = pypownetr.model.PowerNetPyomoModelCambodian(dataset_dir=dataset_dir, year=year)
        pownet_pyomo.net_data.node_lists["h_nodes"] = ["TTYh", "LRCh"]
        pownet_pyomo.net_data.node_lists["gd_nodes"] = ["GS1", "KPCM"]
        pownet_pyomo.net_data.node_lists["gn_nodes"] = ["STH"]
        pownet_pyomo.net_data.node_lists["td_nodes"] = ["BTB", "BMC"]
        pownet_pyomo.net_data.node_lists["tn_nodes"] = ["IE", "KPCG", "OSM", "PRST"]
    return pownet_pyomo


def pownet_solver(location, year=2016, start_day=1, end_day=2):
    # year: simulation year, start and end of simulation
    # start_day: first day of simulation (1 to end_day-1)
    # end_day: last day of simulation (2 to 365)
    assert location in ["cambodian", "artificial"]
    assert start_day >= 1 and start_day <= 365
    assert end_day > start_day and end_day <= 365

    solver_func = partial(ng.optimizers.CMA, budget=10)
    pownet_pyomo = get_pownet_model(location=location, year=year)
    pyomo_model = pownet_pyomo.create_model()
    model_data_path = pownet_pyomo.get_data_path()
    print(f"Save the transformed model data to {model_data_path}")
    answers = solve_pownet_subproblems(
        pyomo_model, model_data_path, solver_func=solver_func, start_day=start_day, last_day=end_day
    )
    return answers


def solve_pownet_problem():
    year = 2016
    start_day = 1
    end_day = 2
    pownet_solver(location="cambodian", year=year, start_day=start_day, end_day=end_day)


if __name__ == "__main__":
    solve_pownet_problem()
