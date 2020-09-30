#################################################################
# Author: Alex Doumanoglou (al3x.doum@gmail.com / aldoum@iti.gr)
# Information Technologies Institute - Visual Computing Lab (https://vcl.iti.gr)
# 30 Sept 2020
#################################################################
import typing as tp
from nevergrad.benchmark.perfcap.core import ExperimentManager
from nevergrad.benchmark.experiments import Experiment

def perfcap_experiment(experiment_filename: str, seedg: tp.Iterator[tp.Optional[int]]) -> tp.Iterator[Experiment]:
    return ExperimentManager(experiment_filename, [2000, 4000, 7000], ["Shiwa", "RandomSearch",
                                                                       "RealSpacePSO", "Powell", "DiscreteOnePlusOne",
                                                                       "CMA", "NGO", "TBPSA", "chainCMAPowell", "DE"], seedg).experiments()