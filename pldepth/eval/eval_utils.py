import mlflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pldepth.data.io_utils import Dataset, get_dataset_type_by_name
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.models_meta import ModelType
from pldepth.models.pl_hourglass import EffNetFullyFledged
from pldepth.models.redweb import FeatureFusionLayer, AdaptiveOutputLayer, BottleneckConvLayer
from pldepth.util.env import init_env, get_config


def load_pl_depth_model(model_path, model_type):
    if model_type == ModelType.FULLY_FLEDGED_REDWEB:
        return keras.models.load_model(model_path,
                                       custom_objects={'HourglassNegativeLogLikelihood': HourglassNegativeLogLikelihood,
                                                       'FeatureFusionLayer': FeatureFusionLayer,
                                                       'AdaptiveOutputLayer': AdaptiveOutputLayer,
                                                       'BottleneckConvLayer': BottleneckConvLayer},
                                       compile=False)
    else:
        return keras.models.load_model(model_path, custom_objects={'EffNetFullyFledged': EffNetFullyFledged},
                                       compile=False)


def get_pl_model_type_from_path(pl_depth_path):
    if pl_depth_path.endswith("FFEffNet_model.h5"):
        return ModelType.FULLY_FLEDGED_EFFNET
    else:
        return ModelType.FULLY_FLEDGED_REDWEB


def get_depth_cap_by_dataset(dataset_type):
    if dataset_type == Dataset.IBIMS:
        depth_cap = 50
    elif dataset_type == Dataset.TUM:
        depth_cap = 10
    elif dataset_type == Dataset.DIODE:
        depth_cap = 350
    elif dataset_type == Dataset.SINTEL:
        # A value of 255 effectively corresponds to 72 (due to the way how the ground truth data is stored)
        # depth_cap = 72
        depth_cap = 255
    elif dataset_type == Dataset.HR_WSI:
        #
        depth_cap = 1
    else:
        raise NotImplementedError("Unrecognized dataset type '{}'".format(dataset_type))

    return depth_cap


def get_inverted_by_dataset_and_model(dataset_type, model_name, metric_evaluation=False):
    # Do not invert PLDepth predictions within metric evaluation (linear regression takes care of it)
    if metric_evaluation and model_name.startswith("PLDepth"):
        return False

    if dataset_type not in [Dataset.SINTEL, Dataset.HR_WSI] or metric_evaluation:
        if model_name in ["mc", "dd_kitti", "dd_nyu", "youtube3d", "bts", "bts_orig"]:
            return False
        else:
            return True
    else:
        if model_name in ["mc", "dd_kitti", "dd_nyu", "youtube3d", "bts", "bts_orig"]:
            return True
        else:
            return False


def get_inverted_by_dataset(dataset_type):
    if dataset_type not in [Dataset.SINTEL, Dataset.HR_WSI]:
        return True
    else:
        return False


def get_datasets_to_be_evaluated(config):
    datasets = []
    for key in config["DATASETS"]:
        if config["DATASETS"].getboolean(key):
            datasets.append(get_dataset_type_by_name(key))
    return datasets


def compute_scale_and_shift(prediction, target, dataset_type, shape0=448, shape1=448):
    depth_cap = get_depth_cap_by_dataset(dataset_type)

    prediction = tf.cast(prediction, dtype=tf.float32)
    prediction = tf.reshape(prediction, [-1, shape0, shape1])

    # prediction = tf.expand_dims(prediction, axis=0)
    # target = tf.expand_dims(target, axis=0)
    target = tf.cast(target, dtype=tf.float32)
    target = tf.reshape(target, [-1, shape0, shape1])

    mask = tf.ones_like(prediction, dtype=tf.float32)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = tf.reduce_sum(mask * prediction * prediction, (1, 2))
    a_01 = tf.reduce_sum(mask * prediction, (1, 2))
    a_11 = tf.reduce_sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = tf.reduce_sum(mask * prediction * target, (1, 2))
    b_1 = tf.reduce_sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = tf.greater(det, 0.)

    x_0 = (tf.boolean_mask(a_11, valid) * tf.boolean_mask(b_0, valid) - tf.boolean_mask(a_01,
                                                                                        valid) * tf.boolean_mask(
        b_1, valid)) / tf.boolean_mask(det, valid)
    x_1 = (-tf.boolean_mask(a_01, valid) * tf.boolean_mask(b_0, valid) + tf.boolean_mask(a_00,
                                                                                         valid) * tf.boolean_mask(
        b_1, valid)) / tf.boolean_mask(det, valid)

    scale = x_0
    shift = x_1

    prediction_aligned = tf.expand_dims(tf.expand_dims(scale, axis=-1), axis=-1) * prediction + tf.expand_dims(
        tf.expand_dims(shift, axis=-1), axis=-1)
    disparity_cap = 1.0 / depth_cap
    prediction_aligned = tf.where(prediction_aligned < disparity_cap, disparity_cap, prediction_aligned)
    # prediction_depth = 1.0 / prediction_aligned
    return prediction_aligned


def construct_validation_mask_for_tf_dataset(dataset, val_proportion=0.25, seed=42):
    """
    Function constructing a validation mask given a TensorFlow dataset, e.g., to calculate statistics on a separate
    validation set.

    :param dataset: TensorFlow dataset
    :param val_proportion: Proportion (in [0,1]) of the validation set
    :param seed: Seed for randomized operations
    :return: Returns a bool NumPy array storing "True" for elements belonging to the validation set and "False" for
    the others.
    """

    validation_mask = np.zeros(tf.data.experimental.cardinality(dataset))
    num_val_prop = int(val_proportion * len(validation_mask))
    validation_mask[:num_val_prop] = 1.
    np.random.seed(seed)
    np.random.shuffle(validation_mask)
    return validation_mask.astype(np.bool)


def adjust_pred(lr, prediction):
    """
    Function applying the learned linear regression to the given predictions. For instance, this method is used to
    apply the scale- and shift-transformation on the prediction to match the ground truth's scale and shift.

    :param lr: scikit-learn LinearRegression model used for the transformation.
    :param prediction: Numerical predictions in the original scale.
    :return: Returns a NumPy array with the transformed predictions (shape is [-1])
    """
    return np.squeeze(lr.predict(np.reshape(prediction, [-1, 1])))


def normalize_gt(gt, dataset_type):
    depth_cap = get_depth_cap_by_dataset(dataset_type)
    if dataset_type == Dataset.DIODE:
        gt = np.where(np.equal(gt, 0.), depth_cap, gt)
    gt = np.where(np.greater(gt, depth_cap), depth_cap, gt)
    gt = gt / depth_cap

    if dataset_type == Dataset.SINTEL:
        gt = 1. - gt

    return gt, depth_cap


def evaluate_metric_errors(prediction, gt, dataset_type, scale_translation_lr=None, scaler=None,
                           invert_predictions=False):
    gt, depth_cap = normalize_gt(gt, dataset_type)

    prediction = np.reshape(prediction, [-1])

    if scaler is not None:
        transformed_preds = np.reshape(prediction, [-1, 1])
        transformed_preds = scaler.transform(transformed_preds)
    else:
        transformed_preds = np.reshape(prediction, [-1, 1])

    if invert_predictions:
        transformed_preds = 1. / (transformed_preds + 1e-10)

    if scale_translation_lr is None:
        scale_translation_lr = LinearRegression()
        scale_translation_lr = scale_translation_lr.fit(np.reshape(transformed_preds, [-1, 1]), np.reshape(gt, [-1, 1]))

    transformed_preds = adjust_pred(scale_translation_lr, transformed_preds)
    transformed_preds = np.clip(transformed_preds, 0., 1.)

    rmse = np.sqrt(mean_squared_error(np.reshape(gt, [-1, 1]), transformed_preds))
    mae = mean_absolute_error(np.reshape(gt, [-1, 1]), transformed_preds)

    gt = np.reshape(gt, [-1])
    transformed_preds = np.reshape(transformed_preds, [-1])
    abs_rel = np.mean(np.abs((transformed_preds + 1e-10) - (gt + 1e-10)) / (gt + 1e-10))

    max_vals = np.maximum((gt + 1e-10) / (transformed_preds + 1e-10), (transformed_preds + 1e-10) / (gt + 1e-10))
    delta_err = np.sum(np.greater(max_vals, 1.25))
    delta_err = delta_err / np.size(gt)

    return rmse, mae, abs_rel, delta_err


def determine_scale_and_translation(dataset, validation_mask, dataset_type, model=None, predictions=None,
                                    model_type=None):
    num_examples = int(np.sum(validation_mask))
    model_depths = np.zeros([num_examples, 448, 448])
    gts = np.zeros([num_examples, 448, 448])

    ctr = 0
    for idx, elem in enumerate(dataset.as_numpy_iterator()):
        if not validation_mask[idx]:
            continue

        image = elem[0]
        gt = elem[1]

        gt, depth_cap = normalize_gt(gt, dataset_type)

        gts[ctr] = gt

        if model is not None:
            if model_type == ModelType.FULLY_FLEDGED_REDWEB:
                preprocess_input_fn = tf.keras.applications.resnet50.preprocess_input
            else:
                from tensorflow.keras.applications.efficientnet import preprocess_input
                preprocess_input_fn = preprocess_input

            input_img = preprocess_input_fn(image)

            # This is the real code
            prediction = model.predict(np.array([input_img]), batch_size=1)
            model_depths[ctr] = np.squeeze(prediction)
        else:
            prediction = np.squeeze(predictions[idx])
            model_depths[ctr] = prediction

        ctr += 1

    if model is not None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0., 1.))
        model_depths = scaler.fit_transform(np.reshape(model_depths, [-1, 1]))
    else:
        scaler = None

    # Do linear regression
    lr = LinearRegression()
    lr = lr.fit(np.reshape(model_depths, [-1, 1]), np.reshape(gts, [-1, 1]))

    return lr, scaler


def initialize_evaluation_run(use_mlflow=True):
    config = init_env(use_mlflow=use_mlflow)

    eval_config = get_config(config_file_name="conf/eval.ini")
    if use_mlflow:
        mlflow.set_experiment(eval_config["MLFLOW"]["EXP_NAME"])

    model_input_shape = [448, 448, 3]

    return config, model_input_shape, eval_config


def load_external_predictions(path):
    return np.load(path, mmap_mode='r')
