import mlflow
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from pldepth.util.time_utils import get_time_str


def log_parameter_dict(param_dict):
    for key in param_dict:
        mlflow.log_param(key, param_dict[key])


def get_model_checkpoint_path(config, use_mlflow=True):
    save_dir = os.path.join(config["DATA"]["CACHE_PATH_PREFIX"], 'saved_models')
    if use_mlflow:
        return os.path.join(save_dir, mlflow.active_run().info.run_id)
    else:
        return os.path.join(save_dir, get_time_str())


def construct_model_checkpoint_callback(config, model_type, verbosity):
    save_dir = get_model_checkpoint_path(config)
    model_name = 'pldepth_%s_model.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    monitor_loss = 'val_loss'

    return ModelCheckpoint(filepath=filepath, monitor=monitor_loss, verbose=verbosity, save_best_only=True)


def get_tensorboard_path(config):
    return os.path.join(config["LOGGING"]["TENSORBOARD_LOG_DIR"], "{}_{}".format(mlflow.active_run().info.experiment_id,
                                                                                 mlflow.active_run().info.run_id))


def construct_tensorboard_callback(config, dir_name):
    return TensorBoard(log_dir=os.path.join(get_tensorboard_path(config), dir_name), profile_batch=0)
