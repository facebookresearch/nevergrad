# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import typing as tp
from . import core
from . import container
from . import _layering
from . import data as pdata
from . import choice as pchoice


def flatten(
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
                content = flatten(p, with_containers=with_containers, order=order)
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
        subparams = flatten(param, with_containers=False, order=order - 1)
        flat += [(f"#{x}", y) for x, y in subparams]
    if not with_containers:
        flat = [(x, y) for x, y in flat if isinstance(y, (pdata.Data, pchoice.BaseChoice))]
    return flat


def list_data(parameter: core.Parameter) -> tp.List[pdata.Data]:
    """List all the Data instances involved as parameter (not as subparameter/
    endogeneous parameter) in the order they are defined in the standardized data.

    Parameter
    ---------
    parameter: Parameter
        the parameter to inspect
    """
    return [x for _, x in flatten(parameter, order=0) if isinstance(x, pdata.Data)]


class ParameterInfo(tp.NamedTuple):
    """Information about a parameter

    Attributes
    ----------
    deterministic: bool
        whether the function equipped with its instrumentation is deterministic.
        Can be false if the function is not deterministic or if the instrumentation
        contains a softmax.
    continuous: bool
        whether the domain is entirely continuous.
    ordered: bool
        whether all domains and subdomains are ordered.
    arity: int
        number of options for discrete parameters (-1 if continuous)
    """

    deterministic: bool
    continuous: bool
    ordered: bool
    arity: int


def analyze(parameter: core.Parameter) -> ParameterInfo:
    """Analyzes a parameter to provide useful information about it"""
    params = list_data(parameter)
    int_layers = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for x in params]))
    return ParameterInfo(
        deterministic=all(lay.deterministic for lay in int_layers),
        continuous=not any(lay.deterministic for lay in int_layers),
        ordered=all(lay.ordered for lay in int_layers),
        arity=max(
            (lay.arity for lay in int_layers if lay.arity is not None), default=-1
        ),  # only softmax params for now
    )


@contextlib.contextmanager
def deterministic_sampling(parameter: core.Parameter) -> tp.Iterator[None]:
    """Temporarily change the behavior of a Parameter to become deterministic

    Parameters
    ----------
    parameter: Parameter
        the parameter which must behave deterministically during the "with" context
    """
    all_data = list_data(parameter)
    int_layers = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for x in all_data]))
    # record state and set deterministic to True
    deterministic = [lay.deterministic for lay in int_layers]
    for lay in int_layers:
        lay.deterministic = True
    yield
    # sample and reset the previous behavior
    parameter.value  # pylint: disable=pointless-statement
    for lay, det in zip(int_layers, deterministic):
        lay.deterministic = det
