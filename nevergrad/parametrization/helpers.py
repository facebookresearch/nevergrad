# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import random
import typing as tp
import numpy as np
from . import core
from . import container
from . import _layering
from . import data as pdata
from . import choice as pchoice


def flatten_parameter(
    parameter: core.Parameter, with_containers: bool = True, order: int = 0
) -> tp.List[tp.Tuple[str, core.Parameter]]:
    """List all the instances involved as parameter (not as subparameter/
    endogeneous parameter)

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect
    with_containers: bool
        returns all the sub-parameter if True, otherwise only non-pure containers (i.e. only Data and Choice)
    order: int
        order of model/internal parameters to extract. With 0, no model/internal parameters is
        extracted, with 1, only 1st order are extracted, with 2, so model/internal parameters and
        their own model/internal parameters etc. Order 1 subparameters/endogeneous parameters
        include sigma for instance.


    Returns
    -------
    list
        a list of all (name, parameter) implied in this parameter, i.e all choices, items of dict
        and tuples etc (except if only_data=True). Names have a format "<index>.<key>" for a tuple
        containing dicts containing data for instance. Supbaramaters have # in their names.
        The parameters are sorted in the same way they would appear in the standard data.

    """
    flat = [("", parameter)]
    if isinstance(parameter, container.Container):
        content_to_add: tp.List[container.Container] = [parameter]
        if isinstance(parameter, container.Instrumentation):  # special case: skip internal Tuple and Dict
            content_to_add = [parameter[0], parameter[1]]  # type: ignore
        for c in content_to_add:
            for k, p in sorted(c._content.items()):
                content = flatten_parameter(p, with_containers=with_containers, order=order)
                flat.extend(
                    (str(k) + ("" if not x else ("." if not x.startswith("#") else "") + x), y)
                    for x, y in content
                )
    if order > 0 and not isinstance(parameter, container.Container):
        subcontent = dict(parameter._subobjects.items())
        param = container.Dict(**subcontent)
        if len(subcontent) == 1:
            lone_content = next(iter(subcontent.values()))
            if isinstance(lone_content, container.Dict):
                param = lone_content  # shorcut subparameters
        subparams = flatten_parameter(param, with_containers=False, order=order - 1)
        flat += [(f"#{x}", y) for x, y in subparams]
    if not with_containers:
        flat = [(x, y) for x, y in flat if isinstance(y, (pdata.Data, pchoice.BaseChoice))]
    return flat


def list_data(parameter: core.Parameter) -> tp.List[tp.Tuple[str, pdata.Data]]:
    return [x for x in flatten_parameter(parameter, order=0) if isinstance(x[1], pdata.Data)]  # type: ignore


@contextlib.contextmanager
def deterministic_sampling(parameter: core.Parameter) -> tp.Iterator[None]:
    """Temporarily change the behavior of a Parameter to become deterministic

    Parameters
    ----------
    parameter: Parameter
        the parameter which must behave deterministically during the "with" context
    """
    all_params = list_data(parameter)
    int_layers = list(itertools.chain.from_iterable([_layering.Int.filter_from(x[1]) for x in all_params]))
    # record state and set deterministic to True
    deterministic = [lay.deterministic for lay in int_layers]
    for lay in int_layers:
        lay.deterministic = True
    yield
    # sample and reset the previous behavior
    parameter.value  # pylint: disable=pointless-statement
    for lay, det in zip(int_layers, deterministic):
        lay.deterministic = det


#     @classmethod
#     def list_arrays(cls, parameter: p.Parameter) -> tp.List[p.Data]:
#         """Computes a list of Data (Array/Scalar) parameters in the same order as in
#         the standardized data space.
#         """
#         if isinstance(parameter, p.Data):
#             return [parameter]
#         elif isinstance(parameter, p.Constant):
#             return []
#         if not isinstance(parameter, p.Container):
#             raise RuntimeError(f"Unsupported parameter {parameter}")
#         output: tp.List[p.Data] = []
#         for _, subpar in sorted(parameter._content.items()):
#             output += cls.list_arrays(subpar)
#         return output

# pylint: disable=too-many-locals


def split_as_data_parameters(
    parameter: core.Parameter,
) -> tp.List[tp.Tuple[str, pdata.Data]]:
    """List all the instances involved as parameter (not as subparameter/
    endogeneous parameter)

    Parameter
    ---------
    parameter: Parameter
        the parameter to split

    Returns
    -------
    list
        the list and subparameters ordered as in data space

    Note
    ----
    This function is experimental, its output will probably evolve before converging.
    """
    err_msg = (
        f"Could not figure out the data order for: {parameter} "
        "(please open an issue on nevergrad github repository)"
    )
    copied = parameter.copy()
    ref = parameter.copy()
    flatp, flatc, flatref = (dict(list_data(pa)) for pa in (parameter, copied, ref))
    keys = list(flatp.keys())
    random.shuffle(keys)  # makes it safer to test!
    # remove transforms for both ref and copied parameters and set index
    for k, key in enumerate(keys):
        for not_ref, flat in enumerate((flatref, flatc)):
            param = flat[key]
            param._layers = param._layers[:1]  # force remove the bound to avoid weird clipping etc
            param.set_mutation(sigma=1.0)  # force mutation sigma to 1 to avoid rounding
            if not_ref:
                param.set_standardized_data(k * np.ones((param.dimension)))
    # analyze results
    data = copied.get_standardized_data(reference=ref)
    order: tp.List[int] = []
    for val, _ in itertools.groupby(data):
        num = int(np.round(val))
        if num in order:
            if order[-1] != num:
                raise RuntimeError(err_msg)
        else:
            order.append(num)
    if len(order) != len(flatp):
        raise RuntimeError(err_msg)
    # create output and check it
    ordered_keys = [keys[num] for num in order]
    ordered_arrays = [(k, flatp[k]) for k in ordered_keys]
    # print(f"DEBUGGING:\nkeys={keys}\ndata={data}\norder={order}\nordered_key={ordered_keys}")
    if sum(pa.dimension for _, pa in ordered_arrays) != parameter.dimension:
        raise RuntimeError(err_msg)
    return ordered_arrays
