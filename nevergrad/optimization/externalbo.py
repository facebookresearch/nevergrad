# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import base
from .base import IntOrParameter
import nevergrad.common.typing as tp
from nevergrad.parametrization import transforms
from nevergrad.parametrization import parameter as p

import hyperopt  # type: ignore
from hyperopt import hp, Trials, Domain, tpe  # type: ignore


def _hp_parametrization_to_dict(x, **kwargs):
    if isinstance(x, p.Instrumentation):
        x_dict = kwargs.get("default", {})
        for idx_param in range(len(x[0].value)):
            x_dict.update(_hp_parametrization_to_dict(x[0][idx_param], name=str(idx_param)))
        for name in x[1].value.keys():
            x_dict.update(_hp_parametrization_to_dict(x[1][name], name=name))
        return x_dict
    elif isinstance(x, (p.Log, p.Scalar)):
        return {kwargs["name"]: [x.value]}
    elif isinstance(x, p.Choice):
        x_dict = {}
        for i in range(len(x.choices)):
            if x.value == x.choices[i].value:
                x_dict[kwargs["name"]] = [i]
                if isinstance(x.choices[i], (p.Log, p.Scalar, p.Choice)):
                    x_dict[kwargs["name"] + f"__{i}"] = [x.choices[i].value]
                elif isinstance(x.choices[i], p.Instrumentation):
                    x_dict.update(_hp_parametrization_to_dict(x.choices[i]))
        return x_dict


def _hp_dict_to_parametrization(x):
    if isinstance(x, dict) and ("args" in x) and ("kwargs" in x):
        x["args"] = tuple([_hp_dict_to_parametrization(x["args"][str(i)]) for i in range(len(x["args"]))])
        x["kwargs"] = {k: _hp_dict_to_parametrization(v) for k, v in x["kwargs"].items()}
        return (x["args"], x["kwargs"])
    return x


def _get_search_space(param_name, param):
    if isinstance(param, p.Instrumentation):
        space = {}
        space["args"] = {
            str(idx_param): _get_search_space(str(idx_param), param[0][idx_param])  # type: ignore
            for idx_param in range(len(param[0].value))
        }
        space["kwargs"] = {
            param_name: _get_search_space(param_name, param[1][param_name])  # type: ignore
            for param_name in param[1].value.keys()
        }
        return space

    elif isinstance(param, (p.Log, p.Scalar)):
        if (param.bounds[0][0] is None) or (param.bounds[1][0] is None):
            if isinstance(param, p.Scalar) and not param.integer:
                return hp.lognormal(label=param_name, mu=0, sigma=1)
            raise ValueError(f"Scalar {param_name} not bounded.")
        elif isinstance(param, p.Log):
            return hp.loguniform(
                label=param_name, low=np.log(param.bounds[0][0]), high=np.log(param.bounds[1][0])
            )
        elif isinstance(param, p.Scalar):
            if param.integer:
                return hp.randint(label=param_name, low=int(param.bounds[0][0]), high=int(param.bounds[1][0]))
            else:
                return hp.uniform(label=param_name, low=param.bounds[0][0], high=param.bounds[1][0])

    elif isinstance(param, p.Choice):
        list_types = [
            type(param.choices[i])
            for i in range(len(param.choices))
            if not isinstance(param.choices[i], (p.Instrumentation, p.Constant))
        ]

        if len(list_types) != len(set(list_types)):
            raise NotImplementedError
        return hp.choice(
            param_name,
            [
                _get_search_space(param_name + "__" + str(i), param.choices[i])
                for i in range(len(param.choices))
            ],
        )

    elif isinstance(param, p.Constant):
        return param.value

    # Hyperopt do not support array
    raise NotImplementedError


class _HyperOpt(base.Optimizer):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        prior_weight: float = 1.0,
        n_startup_jobs: int = 20,
        n_EI_candidates: int = 24,
        gamma: float = 0.25,
        verbose: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        try:
            # try to convert parametrization to hyperopt search space
            if not isinstance(self.parametrization, p.Instrumentation):
                raise NotImplementedError
            self.space = _get_search_space(self.parametrization.name, self.parametrization)
            self._transform = None
        except NotImplementedError:
            self._transform = transforms.ArctanBound(0, 1)
            self.space = {f"x_{i}": hp.uniform(f"x_{i}", 0, 1) for i in range(self.dimension)}

        self.trials = Trials()
        self.domain = Domain(fn=None, expr=self.space, pass_expr_memo_ctrl=False)
        self.tpe_args = {
            "prior_weight": prior_weight,
            "n_startup_jobs": n_startup_jobs,
            "n_EI_candidates": n_EI_candidates,
            "gamma": gamma,
            "verbose": verbose,
        }

    def _internal_ask_candidate(self) -> p.Parameter:
        # Inspired from FMinIter class (hyperopt)
        next_id = self.trials.new_trial_ids(1)
        new_trial = tpe.suggest(
            next_id, self.domain, self.trials, self._rng.randint(2 ** 31 - 1), **self.tpe_args
        )[0]
        self.trials.insert_trial_doc(new_trial)
        self.trials.refresh()

        candidate = self.parametrization.spawn_child()

        if self._transform:
            data = np.array([new_trial["misc"]["vals"][f"x_{i}"][0] for i in range(self.dimension)])
            candidate = candidate.set_standardized_data(self._transform.backward(data))

            # For consistency, we need to update hyperopt history
            # when standardized data is changed
            if any(
                data
                != self._transform.forward(candidate.get_standardized_data(reference=self.parametrization))
            ):
                for it, val in enumerate(
                    self._transform.forward(candidate.get_standardized_data(reference=self.parametrization))
                ):
                    self.trials._dynamic_trials[next_id[0]]["misc"]["vals"][f"x_{it}"][0] = val
        else:
            spec = hyperopt.base.spec_from_misc(new_trial["misc"])
            config = hyperopt.space_eval(self.space, spec)
            candidate.value = _hp_dict_to_parametrization(config)

        candidate._meta["trial_id"] = new_trial["tid"]
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        result = {"loss": loss, "status": "ok"}
        assert "trial_id" in candidate._meta
        tid = candidate._meta["trial_id"]
        assert self.trials._dynamic_trials[tid]["state"] == hyperopt.JOB_STATE_NEW

        now = hyperopt.utils.coarse_utcnow()
        self.trials._dynamic_trials[tid]["book_time"] = now
        self.trials._dynamic_trials[tid]["refresh_time"] = now
        self.trials._dynamic_trials[tid]["state"] = hyperopt.JOB_STATE_DONE
        self.trials._dynamic_trials[tid]["result"] = result
        self.trials._dynamic_trials[tid]["refresh_time"] = hyperopt.utils.coarse_utcnow()
        self.trials.refresh()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        next_id = self.trials.new_trial_ids(1)
        new_trial = hyperopt.rand.suggest(next_id, self.domain, self.trials, self._rng.randint(2 ** 31 - 1))
        self.trials.insert_trial_docs(new_trial)
        self.trials.refresh()
        tid = next_id[0]

        if self._transform:
            data = candidate.get_standardized_data(reference=self.parametrization)
            data = self._transform.forward(data)
            self.trials._dynamic_trials[tid]["misc"]["vals"] = {f"x_{i}": [data[i]] for i in range(len(data))}
        else:
            null_config: dict = {k: [] for k in self.trials._dynamic_trials[tid]["misc"]["vals"].keys()}
            new_vals: dict = _hp_parametrization_to_dict(candidate, default=null_config)
            self.trials._dynamic_trials[tid]["misc"]["vals"] = new_vals

        self.trials.refresh()
        candidate._meta["trial_id"] = tid
        self._internal_tell_candidate(candidate, loss)


class ParametrizedHyperOpt(base.ConfiguredOptimizer):
    # pylint: disable=too-many-instance-attributes
    """Hyperopt: Distributed Asynchronous Hyper-parameter Optimization.
    This class is a wrapper over the `hyperopt <https://github.com/hyperopt/hyperopt>`_ package.

    Parameters
    ----------
    parametrization: int or Parameter
        Parametrization object
    budget: int
        Number of iterations
    num_workers: int
        Number of workers
    prior_weight: float (default 1.0)
        Smoothing factor to avoid having zero probabilities
    n_startup_jobs: int (default 20)
        Number of random uniform suggestions at initialization
    n_EI_candidates: int (default 24)
        Number of generated candidates during EI maximization
    gamma: float (default 0.25)
        Threshold to split between l(x) and g(x), see eq. 2 in

    verbose: bool (default False)
        Hyperopt algorithm verbosity

    Note
    ----
    HyperOpt is described in Bergstra, James S., et al.
    "Algorithms for hyper-parameter optimization."
    Advances in neural information processing systems. 2011
    """

    no_parallelization = False

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        prior_weight: float = 1.0,
        n_startup_jobs: int = 20,
        n_EI_candidates: int = 24,
        gamma: float = 0.25,
        verbose: bool = False,
    ) -> None:
        super().__init__(_HyperOpt, locals())


HyperOpt = ParametrizedHyperOpt().set_name("HyperOpt", register=True)
