# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from . import core
from . import container
from . import choice


def flatten_parameter(
        parameter: core.Parameter,
        with_containers: bool = True,
        order: int = 0
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
    if isinstance(parameter, core.Dict):
        content_to_add: tp.List[core.Dict] = [parameter]
        if isinstance(parameter, container.Instrumentation):  # special case: skip internal Tuple and Dict
            content_to_add = [parameter[0], parameter[1]]  # type: ignore
        for c in content_to_add:
            for k, p in c._content.items():
                content = flatten_parameter(p, with_containers=with_containers, order=order)
                flat.update({str(k) + ("" if not x else ("." if not x.startswith("#") else "") + x): y for x, y in content.items()})
    if order > 0 and parameter._parameters is not None:
        subparams = flatten_parameter(parameter.parameters, with_containers=False, order=order - 1)
        flat.update({"#" + str(x): y for x, y in subparams.items()})
    if not with_containers:
        flat = {x: y for x, y in flat.items() if not isinstance(y, (core.Dict, core.Constant)) or isinstance(y, choice.BaseChoice)}
    return flat
