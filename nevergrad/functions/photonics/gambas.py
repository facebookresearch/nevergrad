# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import typing as tp
from .mode_converter import mode_converter
from .bender import bender


first_time_ceviche = True


def gambas_function(
    x: np.ndarray, benchmark_type: int = 0, discretize=False, wantgrad=False, wantfields=False
) -> tp.Any:
    """
    x = 2d or 3d array of scalars
     Inputs:
    1. benchmark_type = 0 1 2 or 3, depending on which benchmark you want
    2. discretize = True if we want to force x to be in {0,1} (just checks sign(x-0.5) )
    3. wantgrad = True if we want to know the gradient
    4. wantfields = True if we want the fields of the simulation
     Returns:
    1. the loss (to be minimized)
    2. the gradient or none (depending on wantgrad)
    3. the fields or none (depending on wantfields
    """
    global first_time_ceviche
    global model
    import autograd  # type: ignore

    #    import autograd.numpy as npa  # type: ignore
    #    import ceviche_challenges  # type: ignore
    #    import autograd  # type: ignore

    # ceviche_challenges.beam_splitter.prefabs
    # ceviche_challenges.mode_converter.prefabs
    # ceviche_challenges.waveguide_bend.prefabs
    # ceviche_challenges.wdm.prefabs

    if benchmark_type == 2:  # TODO
        design_variable_shape = (30, 30)
        simulate = mode_converter
    elif benchmark_type == 0:
        design_variable_shape = (45, 45)
        simulate = bender
    if first_time_ceviche or x is None:
        if benchmark_type == 2:
            design_variable_shape = (30, 30)
            simulate = mode_converter
        elif benchmark_type == 0:
            design_variable_shape = (45, 45)
            simulate = bender
        else:
            print("Benchmark not defined yet")
            return None
        # elif benchmark_type == 1:
        #     spec = ceviche_challenges.beam_splitter.prefabs.pico_splitter_spec()
        #     params = ceviche_challenges.beam_splitter.prefabs.pico_splitter_sim_params()
        #     model = ceviche_challenges.beam_splitter.model.BeamSplitterModel(params, spec)
        # elif benchmark_type == 3:
        #     spec = ceviche_challenges.wdm.prefabs.wdm_spec()
        #     params = ceviche_challenges.wdm.prefabs.wdm_sim_params()
        #     model = ceviche_challenges.wdm.model.WdmModel(params, spec)
    if discretize:
        # x between 0 and 1 shifted to either 0 or 1
        x = np.round(x)
    design = x

    if isinstance(x, str) and x == "name":
        return {0: "waveguide-bend", 1: "beam-splitter", 2: "mode-converter", 3: "wdm"}[benchmark_type]
    elif x is None:
        return design_variable_shape

    assert x.shape == design_variable_shape, f"Expected shape: {design_variable_shape}"  # type: ignore

    # The model class provides a convenience property, `design_variable_shape`
    # which specifies the design shape it expects.

    # Construct a loss function, assuming the `model` and `design` from the code
    # snippet above are instantiated.

    def loss_fn(x, fields_are_needed=False):
        """A simple loss function"""
        the_loss, field = simulate(x.reshape(design_variable_shape))

        if fields_are_needed:
            return the_loss, field
        return the_loss

    def loss_fn_nograd(x, fields_are_needed=False):
        """A simple loss function"""
        the_loss, field = simulate(x.reshape(design_variable_shape))

        if fields_are_needed:
            return the_loss, field
        return the_loss

    if wantgrad:
        loss_value_npa, grad = autograd.value_and_grad(loss_fn)(design)  # type: ignore
    else:
        loss_value = loss_fn_nograd(design)  # type: ignore
        # print("DIFF:", loss_value, loss_value_npa)
    # loss_value, loss_grad = autograd.value_and_grad(loss_fn)(design)  # type: ignore
    first_time_ceviche = False
    assert not (wantgrad and wantfields)
    if wantgrad:
        # if loss_value_npa< 0.0:
        #    print(x, "NP:", loss_fn_nograd(x), "NPA:", loss_fn(x))
        #    assert False, f"superweird_{loss_fn_nograd(x)}_{loss_fn(x)}"
        return loss_value_npa, grad
    if wantfields:
        loss_value, fields = loss_fn_nograd(design, fields_are_needed=True)
        # if loss_value < 0.0:
        #    print(x, "NP:", loss_fn_nograd(x), "NPA:", loss_fn(x))
        #    assert False, f"superweird_{loss_fn_nograd(x)}_{loss_fn(x)}"
        return loss_value, fields
    # if loss_value < 0.0:
    #     print(x, "NP:", loss_fn_nograd(x), "NPA:", loss_fn(x))
    #     assert False, f"superweird_{loss_fn_nograd(x)}_{loss_fn(x)}"
    return loss_value


if __name__ == "__main__":
    design_variable_shape = (30, 30)
    x = np.random.random(design_variable_shape)
    loss = gambas_function(x, benchmark_type=2, discretize=False)
    print(loss)
