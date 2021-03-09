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
from . import choice
from . import _layering
from . import data as pdata


def flatten_parameter(
    parameter: core.Parameter, with_containers: bool = True, order: int = 0
) -> tp.Dict[str, core.Parameter]:
    """List all the instances involved as parameter (not as subparameter/
    endogeneous parameter)

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect
    with_container: bool
        returns only non-container instances (aka no Dict, Tuple, Instrumentation or Constant)
    order: int
        order of model/internal parameters to extract. With 0, no model/internal parameters is
        extracted, with 1, only 1st order are extracted, with 2, so model/internal parameters and
        their own model/internal parameters etc...

    Returns
    -------
    dict
        a dict of all parameters implied in this parameter, i.e all choices, items of dict
        and tuples etc, but not the subparameters/endogeneous parameters like sigma
        with keys if type "<index>.<key>" for a tuple containing dicts containing data for instance.

    Note
    ----
    This function is experimental, its output will probably evolve before converging.
    """
    flat = {"": parameter}
    if isinstance(parameter, container.Container):
        content_to_add: tp.List[container.Container] = [parameter]
        if isinstance(parameter, container.Instrumentation):  # special case: skip internal Tuple and Dict
            content_to_add = [parameter[0], parameter[1]]  # type: ignore
        for c in content_to_add:
            for k, p in c._content.items():
                content = flatten_parameter(p, with_containers=with_containers, order=order)
                flat.update(
                    {
                        str(k) + ("" if not x else ("." if not x.startswith("#") else "") + x): y
                        for x, y in content.items()
                    }
                )
    if order > 0 and not isinstance(parameter, container.Container):
        content = dict(parameter._subobjects.items())
        param = container.Dict(**content)
        if len(content) == 1:
            lone_content = next(iter(content.values()))
            if isinstance(lone_content, container.Dict):
                param = lone_content  # shorcut subparameters
        subparams = flatten_parameter(param, with_containers=False, order=order - 1)
        flat.update({"#" + str(x): y for x, y in subparams.items()})
    if not with_containers:
        flat = {
            x: y
            for x, y in flat.items()
            if not isinstance(y, (container.Container, core.Constant)) or isinstance(y, choice.BaseChoice)
        }
    return flat


@contextlib.contextmanager
def deterministic_sampling(parameter: core.Parameter) -> tp.Iterator[None]:
    """Temporarily change the behavior of a Parameter to become deterministic

    Parameters
    ----------
    parameter: Parameter
        the parameter which must behave deterministically during the "with" context
    """
    all_params = flatten_parameter(parameter)
    int_layers = list(
        itertools.chain.from_iterable([_layering.Int.filter_from(x) for x in all_params.values()])
    )
    # record state and set deterministic to True
    deterministic = [lay.deterministic for lay in int_layers]
    for lay in int_layers:
        lay.deterministic = True
    yield
    # sample and reset the previous behavior
    parameter.value  # pylint: disable=pointless-statement
    for lay, det in zip(int_layers, deterministic):
        lay.deterministic = det


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
    flatp, flatc, flatref = (
        {x: y for x, y in flatten_parameter(pa).items() if isinstance(y, pdata.Data)}
        for pa in (parameter, copied, ref)
    )
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
