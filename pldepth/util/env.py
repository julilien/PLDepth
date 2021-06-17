import configparser
import os
import mlflow
import mlflow.tensorflow
import logging
import tensorflow as tf

from pldepth.util.time_utils import get_curr_date_str

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../.."
CONFIG_FILE = 'conf/run.ini'
CONFIG_FILE_ENCODING = 'utf-8-sig'


def get_config(path=None, config_file_name=None):
    config = configparser.ConfigParser()

    if config_file_name is None:
        config_file_name = CONFIG_FILE

    if path is None:
        path = os.path.join(ROOT_DIR, config_file_name)
    config.read(path, encoding=CONFIG_FILE_ENCODING)
    return config


def init_mlflow(config, tracking_uri=None, experiment_name=None):
    if tracking_uri is None:
        tracking_uri = config["MLFLOW"]["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name is None:
        today_str = get_curr_date_str()
        experiment_name = "{}_{}".format(today_str, config["MLFLOW"]["MLFLOW_EXP_PREFIX"])
    mlflow.set_experiment(experiment_name)


def init_tensorflow(seed, use_float16=False, num_threads=8):
    if use_float16:
        import keras.backend as keras_backend

        dtype = 'float16'
        keras_backend.set_floatx(dtype)
        keras_backend.set_epsilon(1e-4)

    # Solves an issue with regard to the use of newest CUDA versions
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    _ = InteractiveSession(config=config)

    tf.random.set_seed(seed)

    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)


def init_env(tracking_uri=None, experiment_name=None, autolog_freq=100, seed=0, use_float16=False, use_mlflow=True):
    config = get_config()
    if use_mlflow:
        init_mlflow(config, tracking_uri, experiment_name)

    if config["LOGGING"]["LOG_LEVEL"] == "DEBUG":
        log_level = logging.DEBUG
    elif config["LOGGING"]["LOG_LEVEL"] == "INFO":
        log_level = logging.INFO
    elif config["LOGGING"]["LOG_LEVEL"] == "WARNING":
        log_level = logging.WARNING
    elif config["LOGGING"]["LOG_LEVEL"] == "ERROR":
        log_level = logging.ERROR
    else:
        raise ValueError(
            "Unknown log level provided in the configuration file: {}".format(config["LOGGING"]["LOG_LEVEL"]))
    logging.basicConfig(level=log_level)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)

    init_tensorflow(seed, use_float16=use_float16)

    if use_mlflow:
        # Enable auto-logging to MLflow to capture TensorBoard metrics
        mlflow.tensorflow.autolog(every_n_iter=autolog_freq)

    return config
