import json
import os
import pathlib
import typing as tp
import logging
import itertools
import copy
import nevergrad as ng
from uuid import uuid4 as uuid_gen
from enum import Enum
from .rmqservice import RMQService as _RMQService, RMQSettings as _RMQSettings
from ...functions.base import ExperimentFunction

class Status(Enum):
    OK = 0
    Failed = 1

class VarNormalizer:
    """
    Normalizes parameter values to range [-1.0,1.0] based on bounds
    This may improve optimizer convergence when scaling is different across parameters (i.e root joint translation vs joints rotation)
    Can be silently deactivated if normalize is set to False in the constructor.
    """

    def __init__(self, lower_bound: float, upper_bound: float, normalize: bool = False):
        """
        lower_bound: variable lower bound
        upper_bound: variable upper bound
        normalize: if False VarNormalizer is deactivated. If True, it will normalize input values in [-1.0,1.0]
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._normalize = normalize

    def denormalize(self, val: float) -> float:
        """
        Denormalize the parameter val
        val: value to denormalize

        Returns denormalized value
        """
        if not self._normalize:
            return val
        else:
            return (val+1)*(self._upper_bound-self._lower_bound)/2.0 + self._lower_bound

    def normalize(self, val: float) -> float:
        """
        Normalize the parameter val
        val: value to normalize

        Returns normalized value
        """
        if not self._normalize:
            return val
        else:
            return 2.0 * (val - self.lower_bound) / (self._upper_bound - self._lower_bound) - 1.0

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def active_lower_bound(self):
        """
        effective lower variable bound
        """
        if self._normalize:
            return -1.0
        else:
            return self._lower_bound

    @property
    def active_upper_bound(self):
        """
        effective upper variable bound
        """

        if self._normalize:
            return 1.0
        else:
            return self._upper_bound

    @property
    def active(self):
        """
        Returns True, if VarNormalizer is active
        """
        return self._normalize

# pylint:disable=too-many-instance-attributes

class Perfcap3DFunction(ExperimentFunction):
    """
    Performance Capture Objective Function. This appears as a blocking call to the caller.
    However, under the hood, it does a network request via Perfcap3DServerComm's RMQService to
    the Perfcap Benchmark Server to get the function evaluation at the specified point
    """
    # pylint:disable=too-many-locals

    def _create_parameters(self, cfg: tp.Dict[str, tp.Any]) -> tp.List[ng.p.Scalar]:
        """
        Creates the parameters for the objective function based on configuration read from experimentX.json
        Objective Function parameters (i.e variables) are based on the Degrees of Freadom (DoFs) of each template mesh joint
        """
        variables = cfg["variables"]
        jids = list(map(lambda x: x["joint_id"], variables))
        jidset = set(jids)
        assert len(jids) == len(jidset)
        for v in variables:
            # joint rotation exponential map has at most 3 degrees of freedom (DOF). Each DOF corresponds to one potential parameter.
            # ["DOFs"] is a boolean list of length 3, indicating which one of the joint axis
            # are considered for rotation. We can disable degrees of freedom per joint by
            # inserting False to the related position index inside this list
            # mutation_variances/bounds_min/bounds_max/mutable_mutation_variances
            # are arrays of lenth equal to the number of DOFs of the joint (i.e number of True values in ["DOFs"] list)
            assert len(v["DOFs"]) == 3
            nvars = sum(map(lambda x: 1 if x else 0, v["DOFs"]))

            assert len(v["mutation_variances"]) == nvars
            assert len(v["bounds_min"]) == nvars
            assert len(v["bounds_max"]) == nvars
            assert len(v["mutable_mutation_variances"]) == nvars

        initial_pose = self._initial_pose
        dofs_flattened = list(itertools.chain(*map(lambda x: x["DOFs"], variables)))
        jids_flattened = list(itertools.chain(*map(lambda x: [x["joint_id"]]*len(list(filter(None, x["DOFs"]))), variables)))

        var_indices_flattened = list(range(len(jids_flattened)))
        dof_indices_flattened = list(itertools.chain(*map(lambda x: [index for index, v in enumerate(x["DOFs"]) if v], variables)))
        self._variable_to_dof_index_map = dict(zip(var_indices_flattened, dof_indices_flattened))
        self._variable_to_joint_index_map = dict(zip(var_indices_flattened, jids_flattened))
        initial_values = list(itertools.compress(itertools.chain(*(initial_pose[jid] for jid in jids)), dofs_flattened))

        # initial_values

        bounds_min = list(itertools.chain(*map(lambda x: x["bounds_min"], variables)))
        bounds_max = list(itertools.chain(*map(lambda x: x["bounds_max"], variables)))
        mutation_variances = list(itertools.chain(*map(lambda x: x["mutation_variances"], variables)))
        mutable_mut_var = list(itertools.chain(*map(lambda x: x["mutable_mutation_variances"], variables)))
        # parameter normalizers, one for each parameter
        self._var_normalizers = [VarNormalizer(lower_bound=x[0], upper_bound=x[1], normalize=self._normalize)
                                 for x in zip(bounds_min, bounds_max)]
        # parameter creation (initial value, bounds, mutable_mutation_var)
        params = [ng.p.Scalar(init=n.normalize(init), lower=n.active_lower_bound, upper=n.active_upper_bound, mutable_sigma=ms)
             for (n, ms, init) in zip(self._var_normalizers, mutable_mut_var, initial_values)]
        # leave negative mutation variances to auto / ie. do not set
        for (v, mv, n) in filter(lambda x: x[1] > 0.0, zip(params, mutation_variances, self._var_normalizers)):
            v.set_mutation(n.normalize(mv), custom="gaussian")      # set variable mutation

        return params

    def __init__(self, experiment_filename : str, **kwargs):

        cfg = Perfcap3DExperimentConfig(experiment_filename)
        objective_func_cfg = cfg.objective_function_config

        self._initial_pose: tp.Dict[int, tp.List[float]] = dict(map(lambda o: (o["joint_id"], o["values"]),
                                                                    objective_func_cfg["initial_pose"]))

        self._relative_search_space: bool = objective_func_cfg["relative_search_space"]
        self._normalize: bool = objective_func_cfg["normalize"]
        self._var_normalizers: tp.List[VarNormalizer] = None
        self._variable_to_dof_index_map: tp.Dict[int, int] = None
        self._variable_to_joint_index_map: tp.Dict[int, int] = None
        self._experiment_tag_id: str = str(uuid_gen()) if 'experiment_tag_id' not in kwargs else kwargs['experiment_tag_id']
        self._server_comm = Perfcap3DServerComm(cfg.server_config)
        self._experiment_started : bool = False
        params = self._create_parameters(objective_func_cfg)
        super().__init__(self.oracle_call, ng.p.Instrumentation(*params))
        self.register_initialization(experiment_filename = experiment_filename,
                                    experiment_tag_id = self._experiment_tag_id)  # to create equivalent instances through "copy"

        # exclude experiment_tag_id from fight descriptors by encapsulating its name inside "{ }"
        # append experiment_tag_id to ExperimentFunction descritors.
        # put name inside "{}", to exclude from fightplots (plotting.py has been updated accordingly)
        self._descriptors[r"{experiment_tag_id}"] = self._experiment_tag_id

    def compute_pose(self, *args) -> tp.Dict[int, tp.List[float]]:
        """
        This function will translate Optimizer's parameter arguments to animatable mesh pose.
        """
        eval_point = copy.deepcopy(self._initial_pose)  # deep copy         # start from initial_pose

        # DOF values that are not set to True in configuration are not touched from the optimizer's evaluation point
        # and are left to the value of the initial pose
        for (v, n, var_index) in zip(args, self._var_normalizers, range(len(args))):
            jid = self._variable_to_joint_index_map[var_index]
            vi = self._variable_to_dof_index_map[var_index]
            v = n.denormalize(v)
            # update evaluation_point based on values of the parameters
            if self._relative_search_space:
                eval_point[jid][vi] += v
            else:
                eval_point[jid][vi] = v

        return eval_point

    def oracle_call(self, *args):
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """

        if not self._experiment_started:
            self._server_comm.start_experiment(self._experiment_tag_id)
            self._experiment_started = True

        eval_point = self.compute_pose(*args)
        errors = self._server_comm.request_function_evaluation(eval_point, self._experiment_tag_id)

        return errors["total"]

    def evaluation_function(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        """
        This is called upon benchmark finish to obtain recommendation value
        and notify benchmark server for the end of the experiment to flush logs
        """

        # pylint: disable=not-callable
        loss = self.function(*args)             # the last optimizer iteration in the server logs
                                                # corresponds to this evaluation call
        assert isinstance(loss, float)

        self._server_comm.stop_experiment(self._experiment_tag_id,
                                                self.compute_pose(*args))
                                                # this notifies server to flush logs with converged pose equal to args.
        return loss


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):

        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# pylint: disable=too-many-instance-attributes

class Perfcap3DServerComm(metaclass = Singleton):
    """
    Perfcap3DServerComm is responsible for all communications to the Perfcap Benchmark Server
    """
    # Perfcap3DFunction class are instantiated with the Perfcap3DServerComm instance injected into it
    # Perfcap3DServerComm uses RMQService to talk with the benchmark server.
    # All communication to the Perfcap Benchmark Server goes through Perfcap3DServerComm

    def _setupRMQ(self) -> _RMQService:

        rmqsettings_fpath = os.path.join(pathlib.Path(os.path.abspath(__file__)).parent, "experiment_config\\rmqsettings.json")
        with open(rmqsettings_fpath) as f:
            rmq_settings = json.load(f)

        # in case Benchmark is run under multi processing, each process (i.e. one instance of Experiment Manager)
        # defines a unique routing_key to receive messages from Perfcap Benchmark Server
        # this routing key is embedded in outgoing messages by RMQService, so that the Perfcap Benchmark Server knows under which
        # routing key to forward the reply.
        rmq_settings = _RMQSettings(conn_str=rmq_settings["uri"], exchange_in=rmq_settings["ng_exchange_in"],
                                    exchange_out=rmq_settings["ng_exchange_out"], routing_key_in=str(uuid_gen()))
        return _RMQService(rmq_settings)

    def __init__(self, server_config : tp.Dict[str,tp.Any]) -> None:
        """
        Perfcap3DServerComm: Communications with the Benchmark Server
        server_config: Dictionary obtained through Perfcap3DExperimentConfig.server_config
                       (contains information for initializing the benchmark server to load the correct dataset
                        and objective function's error term weights)
        """
        self._rmq_service: _RMQService = None
        self._experiment_set: bool = False
        self._no_of_pending_experiments: int = 0
        self._logger = logging.getLogger("Perfcap3DServerComm")
        self._server_cfg: tp.Dict[str, tp.Any] = server_config
        self._experiment_id: int = server_config["experiment_id"]

    def _set_experiment(self) -> None:
        """
        Request Perfcap Benchmark Server to setup an experiment (i.e switch to specific dataset and
        set objective function's error term weights). This function is called by the start_experiment method,
        at the beginning of an experiment. Subsequent calls to set_experiment with the same experiment_id are
        ignored by the Perfcap Benchmark Server. Currently, Perfcap Benchmark Server requires setting the
        experiment once through its life time. Switching to a different experiment (experiment with different experiment_id)
        after a previous experimenthas already been set, is not supported. In this case, consider restarting
        Perfcap Benchmark Server or start a new Perfcap Benchmark Server instance under different RMQSettings
        """

        if self._experiment_set:
            return

        self._logger.info("Setting experiment to: %d", self._experiment_id)
        self._logger.info("Sending Perfcap bechmark server request to set experiment to %d", self._experiment_id)
        msg = self._server_cfg.copy()
        msg.update({"action": "set_experiment"})
        res = self._rmq_service.request_forced(msg)
        try:
            status = Status(res["status"])
            msg = res["message"]

            if status != Status.OK:
                self._logger.error("Benchmark server ERROR status: %s", str(status))
                self._logger.error("Benchmark server ERROR reply: %s", msg)
                raise Exception("Benchmark server ERROR: Could not set experiment to: {0}".format(self._experiment_id))

            self._logger.info("Reply from benchmark server: %s", msg)
            self._logger.info("Experiment %s set successfully", self._experiment_id)
            self._experiment_set = True
        except KeyError as ex:
            self._logger.error("Error while parsing reply for set experiment from benchmark server")
            raise ex

    def _register_experiment(self, experiment_tag_id) -> None:
        """
        Register experiment to benchmark server in order to facilitate logging.
        Optimizer evaluation point history logs and meshes in the converged pose are automatically saved by the benchmark server
        when the experiment is unregistered (i.e at the end of the experiment). Correspondence between nevergrad log and benchmark server
        can be achived via experiment_tag_id

        This Function is called by start_experiment method at the beginning of the experiment
        """
        self._logger.info("Registering experiment with tag id %s", experiment_tag_id)
        msg = {
            "action": "start_experiment",
            "experiment_tag_id": experiment_tag_id
        }
        res = self._rmq_service.request_forced(msg)
        try:
            status = Status(res["status"])
            msg = res["message"]

            if status != Status.OK:
                self._logger.error("Benchmark server ERROR status: %s", str(status))
                self._logger.error("Benchmark server ERROR reply: %s", msg)
                raise Exception("Benchmark server ERROR: Could not register experiment to: {0}".format(experiment_tag_id))

            self._logger.info("Reply from benchmark server: %s", msg)
            self._logger.info("Experiment %s registered successfully", experiment_tag_id)
        except KeyError as ex:
            self._logger.error("Error while parsing reply for register experiment from benchmark server")
            raise ex

    def _unregister_experiment(self, experiment_tag_id, converged_pose: tp.Union[None,tp.Dict[int, tp.List[float]]]) -> None:
        """
        Unregister Experiment, when experiment finishes. This causes benchmark server to flush logs and output meshes
        for this experiment.
        This function is called by stop_experiment method at the end of the experiment
        """
        self._logger.info("Unregistering experiment with tag id %s", experiment_tag_id)
        if converged_pose is not None:
            pose_dict = [{"joint_id": jid, "values": vals} for (jid, vals) in converged_pose.items()]
            msg = {
                "action": "stop_experiment",
                "experiment_tag_id": experiment_tag_id,
                "pose": pose_dict
            }
        else:
            msg = {
                "action": "stop_experiment",
                "experiment_tag_id": experiment_tag_id,
            }
        res = self._rmq_service.request_forced(msg)
        try:
            status = Status(res["status"])
            msg = res["message"]

            if status != Status.OK:
                self._logger.error("Benchmark server ERROR status: %s", str(status))
                self._logger.error("Benchmark server ERROR reply: %s", msg)
                raise Exception("Benchmark server ERROR: Could not unregister experiment: {0}".format(experiment_tag_id))

            self._logger.info("Reply from benchmark server: %s", msg)
            self._logger.info("Experiment %s unregistered successfully", experiment_tag_id)
        except KeyError as ex:
            self._logger.error("Error while parsing reply for unregister experiment from benchmark server")
            raise ex

    def start_experiment(self, experiment_tag_id: str):
        """
        This function is called by Perfcap3DFunction class at the first objective function evaluation.
        It requests Perfcap Benchmark Server to open the project dataset corresponding to the current experiment_id
        and registers the PerfcapExperiment under experiment_tag_id to facilitate logging correspondeces between
        Perfcap Benchmark Server and nevergrad logs.
        """
        self._no_of_pending_experiments += 1
        if self._rmq_service is None:
            self._rmq_service = self._setupRMQ()
        self._set_experiment()
        self._register_experiment(experiment_tag_id)

    def stop_experiment(self, experiment_tag_id: str, converged_pose: tp.Union[None,tp.Dict[int, tp.List[float]]]):
        """
        This will unregister the experiment from Perfcap Benchmark Server, causing any server logging to flush.
        This function is called by Perfcap3DFunction at the end of the experiment, when ExperimentFunction.evaluate_function is called.
        """
        self._unregister_experiment(experiment_tag_id, converged_pose)
        self._no_of_pending_experiments -= 1
        if (self._rmq_service is not None) and (self._no_of_pending_experiments == 0):
            print("Stopping RMQ Service")
            self._rmq_service.stop()

    def request_function_evaluation(self, pose: tp.Dict[int, tp.List[float]], experiment_tag_id: str) -> tp.Dict[str, float]:
        """
        This function will submit a request to the Perfcap Benchmark Server for an objective function evaluation.
        This function is called by Perfcap3DFunction to request objective function evaluations
        """

        pose_dict = [{"joint_id": jid, "values": vals} for (jid, vals) in pose.items()]
        msg = {"action": "function_evaluation",
               "experiment_id": self._experiment_id,
               "experiment_tag_id": experiment_tag_id,
               "pose": pose_dict}

        self._logger.debug("Requesting function evaluation")
        return self._rmq_service.request_forced(msg)

class Perfcap3DExperimentConfig:
    """
    Responsible for reading/loading a perfcap benchmark experiment configuration file
    Each one of the two properties of this class are given as parameters to Perfcap3DServerComm and Perfcap3DFunction constructors,
    respectively
    """
    def _load_experiment_file(self, experiment_cfg_filename) -> tp.Dict[str,tp.Any]:
        """
        Loads an experimentX.json file
        """
        expfpath = os.path.join(pathlib.Path(os.path.abspath(__file__)).parent, "experiment_config", experiment_cfg_filename)
        try:
            with open(expfpath) as f:
                return json.load(f)
        except Exception as ex:
            self._logger.error("Error while loading %s", experiment_cfg_filename)
            self._logger.exception(ex)
            raise ex

    def __init__(self, experiment_cfg_filename : str):

        self._config = self._load_experiment_file(experiment_cfg_filename)

    @property
    def objective_function_config(self):
        return self._config["ng_function_args"]

    @property
    def server_config(self):
        return self._config["benchmark_server_args"]
