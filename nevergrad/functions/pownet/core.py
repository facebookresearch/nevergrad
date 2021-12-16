# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from nevergrad.functions.pyomo.core import Pyomo
from . import solver


class PowNet(Pyomo):
    """Function calling Pyomo model

    Parameters
    ----------
    model: pyomo.environ.model
        Pyomo model constructed from PowNet 2016 data

    Returns
    -------
    float
        the fitness

    Notes
    -----
    - You will require an Pyomo installation (with pip: "pip install pyomo pypownetr")
    - Any changes on the model externally can lead to unexpected behaviours.
    """

    def __init__(self, location: str, day: int = 1) -> None:
        assert location in ["cambodian", "artificial"]
        assert 1 <= day <= 365
        pyomo_model = solver.get_pownet_model(location=location, year=2016)
        _instance = solver.create_pyomo_instance(pyomo_model, day=day)
        super().__init__(model=_instance)
