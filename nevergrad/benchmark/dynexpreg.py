from functools import partial
from .experiments import registry
from .frozenexperiments import perfcap_experiment

# dynamic registration of experiments


def register_perfcap_bench():

    experiment_count = 11
    for i in range(experiment_count):
        experiment_id = "{}".format(i + 1)
        func_name = "perfcap_bench" + experiment_id
        bench_func = partial(perfcap_experiment, experiment_filename="experiment" + experiment_id + ".json")
        registry.register_name(func_name, bench_func)


register_perfcap_bench()
