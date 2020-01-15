import typing as tp
from . import core


def list_parameter_instances(parameter: core.Parameter) -> tp.List[core.Parameter]:
    """List all the instances involved as parameter (not as subparameter/
    endogeneous parameter)

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect

    Returns
    -------
    list
        a list of all parameters implied in this parameter, i.e all choices, items of dict
        and tuples etc, but not the subparameters/endogeneous parameters like sigma
    """
    instances = [parameter]
    if isinstance(parameter, core.Dict):
        for p in parameter._content.values():
            instances += list_parameter_instances(p)
    return instances
