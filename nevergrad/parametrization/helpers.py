# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import contextlib
import itertools
import numpy as np
import nevergrad.common.typing as tp
from . import core
from . import container
from . import _layering
from . import _datalayers
from . import data as pdata
from . import choice as pchoice
from . import transforms as trans


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
    try:
        yield
    finally:
        # sample and reset the previous behavior
        parameter.value  # pylint: disable=pointless-statement
        for lay, det in zip(int_layers, deterministic):
            lay.deterministic = det


def _fully_bounded_layers(data: pdata.Data) -> tp.List[_datalayers.BoundLayer]:
    """Extract fully bounded layers of a Data parameter"""
    layers = _datalayers.BoundLayer.filter_from(data)  # find bound layers
    layers = [  # keep only fully bounded layers
        lay for lay in layers if not any(b is None for b in lay.bounds)
    ]
    return layers


class Normalizer:
    """Hacky way to sample in the space defined by the parametrization.
    Given an vector of values between 0 and 1,
    the transform method samples in the bounds if provided,
    or using the provided function otherwise.
    This is used for samplers.
    Code of parametrization and/or this helper should definitely be
    updated to make it simpler and more robust
    """

    def __init__(
        self,
        reference: core.Parameter,
        unbounded_transform: tp.Optional[trans.Transform] = None,
        only_sampling: bool = False,
    ) -> None:
        self.reference = reference.spawn_child()
        self.reference.freeze()
        # initial check
        parameter = self.reference.spawn_child()
        parameter.set_standardized_data(np.linspace(-1, 1, self.reference.dimension))
        expected = parameter.get_standardized_data(reference=self.reference)
        self._ref_arrays = list_data(self.reference)
        arrays = list_data(parameter)
        check = np.concatenate(
            [x.get_standardized_data(reference=y) for x, y in zip(arrays, self._ref_arrays)], axis=0
        )
        self.working = True
        if not np.allclose(check, expected):
            self.working = False
            self._warn()
        self._only_sampling = only_sampling
        self.unbounded_transform = (
            trans.ArctanBound(0, 1) if unbounded_transform is None else unbounded_transform
        )
        self.fully_bounded = all(bool(_fully_bounded_layers(data)) for data in self._ref_arrays)

    def _warn(self) -> None:
        warnings.warn(
            f"Failed to find bounds for {self.reference}, quasi-random optimizer may be inefficient.\n"
            "Please open an issue on Nevergrad github"
        )

    def backward(self, x: tp.ArrayLike) -> np.ndarray:
        """Transform from [0, 1] to standardized space"""
        return self._apply(x, forward=False)

    def forward(self, x: tp.ArrayLike) -> np.ndarray:
        """Transform from standardized space to [0, 1]"""
        return self._apply(x, forward=True)

    def _apply(self, x: tp.ArrayLike, forward: bool = True) -> np.ndarray:
        utrans = self.unbounded_transform.forward if forward else self.unbounded_transform.backward
        y = np.array(x, copy=True, dtype=float)
        if not self.working:
            return utrans(y)
        try:
            self._apply_unsafe(y, forward=forward)  # inplace
        except Exception:  # pylint: disable=broad-except
            self._warn()
            y = utrans(y)
        return y

    def _apply_unsafe(self, x: np.ndarray, forward: bool = True) -> None:
        # modifies x in place
        start = 0
        utrans = self.unbounded_transform.forward if forward else self.unbounded_transform.backward
        for ref in self._ref_arrays:
            end = start + ref.dimension
            layers = _fully_bounded_layers(ref)
            if self._only_sampling:  # for samplers
                layers = [lay for lay in layers if lay.uniform_sampling]
            if not layers:
                x[start:end] = utrans(x[start:end])
            else:
                layer_index = layers[-1]._layer_index
                array = ref.spawn_child()
                if forward:
                    array.set_standardized_data(x[start:end])
                    x[start:end] = array._layers[layer_index].get_normalized_value()[:]  # type: ignore

                else:
                    normalized = x[start:end].reshape(ref._value.shape)
                    array._layers[layer_index].set_normalized_value(normalized)  # type: ignore
                    x[start:end] = array.get_standardized_data(reference=ref)
            start = end
