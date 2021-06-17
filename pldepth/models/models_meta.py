from enum import Enum
import json
import copy

from pldepth.util.tracking_utils import log_parameter_dict


class StringEnum(Enum):
    def __str__(self):
        return str(self.value)


class ModelType(StringEnum):
    FULLY_FLEDGED_REDWEB = "FFReDWeb"
    FULLY_FLEDGED_EFFNET = "FFEffNet"


def get_model_type_by_name(model_name):
    if model_name == "ff_redweb":
        return ModelType.FULLY_FLEDGED_REDWEB
    elif model_name == "ff_effnet":
        return ModelType.FULLY_FLEDGED_EFFNET
    else:
        raise ValueError("Unknown model name: {}".format(model_name))


class ModelParameters(object):
    def __init__(self):
        self.parameters = {}

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name, default=None):
        if name in self.parameters:
            return self.parameters[name]
        else:
            return default

    def log_parameters(self):
        log_parameter_dict(self.parameters)

    def get_parameter_string(self):
        output_str = ""
        for key in self.parameters:
            if output_str != "":
                output_str += "_"
            output_str += str(key) + "_" + str(self.parameters[key])
        return output_str

    def load_parameters_from_file(self, json_file_path, key, exclude_keys=None):
        with open(json_file_path) as ssfile:
            ext_params = json.load(ssfile)

        if key not in ext_params:
            raise ValueError(
                'Could not find entry for key {} in external parameter file {}.'.format(key, json_file_path))
        else:
            for param_key in ext_params[key]:
                if exclude_keys is not None and param_key in exclude_keys:
                    continue
                value = ext_params[key][param_key]
                if isinstance(value, str):
                    value = value == "True" or value == "true"
                self.set_parameter(param_key, value)

    def duplicate(self):
        result = ModelParameters()
        result.parameters = copy.deepcopy(self.parameters)
        return result
