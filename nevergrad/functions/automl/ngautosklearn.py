import os
import nevergrad as ng
import ConfigSpace as cs

def check_configuration(config_space, values):
    val_dict = to_dict(values[1])
    try:
        config = cs.Configuration(configuration_space=config_space, values=val_dict, allow_inactive_with_values=False)
    except Exception:
        return False
    return True

def get_configuration(values, config_space):
    val_dict = to_dict(values)
    return cs.Configuration(configuration_space=config_space, values=val_dict, allow_inactive_with_values=False)


def get_config_space():
    from ConfigSpace.read_and_write import json as json
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configspace.json')) as f:
        jason_string = f.read()
        config_space = json.read(jason_string)
    return config_space

def get_instrumention(param):
    if param["type"] == "categorical":
        return ng.p.Choice(param["choices"])
    elif param["type"] == "uniform_int":
        if param["log"] == False:
            return ng.p.Scalar(lower=param["lower"], upper=param["upper"], init=param["default"]).set_integer_casting()
        else:
            return ng.p.Log(lower=param["lower"], upper=param["upper"], init=param["default"]).set_integer_casting()
    elif param["type"] == "uniform_float":
        if param["log"] == False:
            return ng.p.Scalar(lower=param["lower"], upper=param["upper"], init=param["default"])
        else:
            return ng.p.Log(lower=param["lower"], upper=param["upper"], init=param["default"])
    elif param["type"] == "constant":
        return ng.p.Constant(param["value"])
    raise Exception(r"{param} type not known")


def get_parametrization():
    import json
    config_space = get_config_space()
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configspace.json')) as f:
        config_space_json = json.load(f)

    base_pipeline = [
        "balancing:strategy",
        "classifier:__choice__",
        "data_preprocessing:categorical_transformer:categorical_encoding:__choice__",
        "data_preprocessing:categorical_transformer:category_coalescence:__choice__",
        "data_preprocessing:numerical_transformer:imputation:strategy",
        "data_preprocessing:numerical_transformer:rescaling:__choice__",
        "feature_preprocessor:__choice__",
    ]

    params = {}

    for param in config_space_json["hyperparameters"]:
        if param["name"] in base_pipeline:
            if param["name"] in ["classifier:__choice__", "feature_preprocessor:__choice__",
                                 "data_preprocessing:numerical_transformer:rescaling:__choice__",
                                 "data_preprocessing:categorical_transformer:category_coalescence:__choice__"]:
                params[param["name"]] = ng.p.Choice([
                    ng.p.Tuple(ng.p.Constant(param_choice), ng.p.Dict(**{
                        hp["name"]: get_instrumention(hp)
                        for hp in config_space_json["hyperparameters"]
                        if param_choice in hp["name"]
                    }))
                    for param_choice in param["choices"]])
            else:
                params[param["name"]] = get_instrumention(param)

    inst = ng.p.Instrumentation(**params)
    from functools import partial
    constraint_check_func = partial(check_configuration, config_space)
    inst.register_cheap_constraint(constraint_check_func)
    return inst, config_space

def to_dict(values):
    clf = values["classifier:__choice__"]
    features = values["feature_preprocessor:__choice__"]
    trans_cat = values["data_preprocessing:categorical_transformer:category_coalescence:__choice__"]
    trans_num = values["data_preprocessing:numerical_transformer:rescaling:__choice__"]
    del values["classifier:__choice__"]
    del values["feature_preprocessor:__choice__"]
    del values["data_preprocessing:categorical_transformer:category_coalescence:__choice__"]
    del values["data_preprocessing:numerical_transformer:rescaling:__choice__"]
    values["classifier:__choice__"] = clf[0]
    values.update(clf[1])
    values["feature_preprocessor:__choice__"] = features[0]
    values.update(features[1])
    values["data_preprocessing:categorical_transformer:category_coalescence:__choice__"] = trans_cat[0]
    if len(trans_cat[1]) > 0: values.update(trans_cat[1])
    values["data_preprocessing:numerical_transformer:rescaling:__choice__"] = trans_num[0]
    if len(trans_num[1]) > 0: values.update(trans_num[1])
    return values