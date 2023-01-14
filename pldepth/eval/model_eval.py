import os
import click

from tensorflow import keras
import logging
import numpy as np

from tensorflow_ranking.python.keras.metrics import DCGMetric, NDCGMetric

from pldepth.data.dao.dao_meta import get_dao_for_dataset_type
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.generic_ranking_provider import GenericHourglassPairRelationDataProvider, \
    GenericHourglassRankingDataProvider
from pldepth.eval.eval_utils import get_inverted_by_dataset, load_external_predictions, \
    get_inverted_by_dataset_and_model, evaluate_metric_errors, initialize_evaluation_run, \
    get_datasets_to_be_evaluated, get_pl_model_type_from_path
from pldepth.eval.metrics import WKDR, WKDR_eq, WKDR_neq
from pldepth.eval.test_suite import TFDatasetTestSuite
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.models_meta import ModelParameters, ModelType
from pldepth.models.pl_hourglass import EffNetFullyFledged
from pldepth.models.redweb import FeatureFusionLayer, AdaptiveOutputLayer, BottleneckConvLayer


def get_default_model_parameters(dataset_type):
    model_params = ModelParameters()
    model_params.set_parameter("val_rankings_per_img", 50000)
    model_params.set_parameter("seed", 0)
    model_params.set_parameter("batch_size", 4)
    model_params.set_parameter("dataset", dataset_type)
    return model_params


def evaluate_model_predictions(config, model_input_shape, dataset_type, model_name, model_type, model=None,
                               external_pred_path=None, invert_ds_relations=False, seed=42, wkdr_threshold=0.03,
                               query_ranking_size=500):
    assert (model is not None and external_pred_path is None) or (
            model is None and external_pred_path is not None), "Either a model or external predictions must be given."

    dao = get_dao_for_dataset_type(dataset_type, config, model_input_shape)
    dataset = dao.get_test_dataset()

    model_params = get_default_model_parameters(dataset_type)

    dataset_inversion = get_inverted_by_dataset(dataset_type)

    generic_relation_provider = GenericHourglassPairRelationDataProvider(model_params, seed=seed,
                                                                         invert_relation_sign=dataset_inversion,
                                                                         threshold=wkdr_threshold,
                                                                         save_pairs_on_disk=True, config=config)
    ordinal_pairs = generic_relation_provider.provide_test_dataset(dataset)

    generic_ranking_provider = GenericHourglassRankingDataProvider(model_params, query_ranking_size=query_ranking_size,
                                                                   seed=seed, invert_relation_sign=dataset_inversion,
                                                                   threshold=wkdr_threshold, save_rankings_on_disk=True,
                                                                   config=config)
    rankings = generic_ranking_provider.provide_test_dataset(dataset)

    if external_pred_path is not None:
        predictions = load_external_predictions(external_pred_path)
    else:
        predictions = None

    metric_inversion = get_inverted_by_dataset_and_model(dataset_type, model_name=model_name, metric_evaluation=True)

    # Assume that only neq-relations are predicted
    pred_wkdr_threshold = 0.0

    pairwise_metrics = [WKDR(wkdr_threshold, pred_wkdr_threshold, invert_ds_relations),
                        WKDR_eq(wkdr_threshold, pred_wkdr_threshold, invert_ds_relations),
                        WKDR_neq(wkdr_threshold, pred_wkdr_threshold, invert_ds_relations)]

    ranking_metrics = [DCGMetric(), NDCGMetric()]

    test_suite = TFDatasetTestSuite(pairwise_metrics, ranking_metrics, dataset_type, model_name, ordinal_pairs,
                                    rankings, seed, query_ranking_size, wkdr_threshold,
                                    invert_rankings_again=(not invert_ds_relations))

    if model_type == ModelType.FULLY_FLEDGED_REDWEB:
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    total_rmse = 0
    total_mae = 0
    total_abs_rel = 0
    total_delta_err = 0
    total_ctr = 0

    max_value_so_far = 0
    mean_value_so_far = 0

    for img_idx, elem in enumerate(test_suite.provide_test_image_iterator()):

        img = elem[0][0]
        gt = elem[0][1]

        if np.max(gt) > max_value_so_far:
            max_value_so_far = np.max(gt)
        mean_value_so_far += np.mean(gt)

        if predictions is not None:
            prediction = np.squeeze(predictions[img_idx])
            raw_prediction = prediction
        else:
            input_img = preprocess_input(img)

            # This is the real code
            raw_prediction = model.predict(np.array([input_img]), batch_size=1)
            exp_prediction = np.exp(raw_prediction)
            prediction = np.squeeze(exp_prediction)

        rmse, mae, abs_rel, delta_err = evaluate_metric_errors(raw_prediction, gt, dataset_type,
                                                               invert_predictions=metric_inversion)

        total_rmse += rmse
        total_mae += mae
        total_abs_rel += abs_rel
        total_delta_err += delta_err
        total_ctr += 1

        if np.min(prediction) < 0.:
            prediction = np.exp(prediction)

        test_suite.submit_prediction(img_idx, prediction)

    rmse_result = total_rmse / total_ctr
    mae_result = total_mae / total_ctr
    abs_rel_result = total_abs_rel / total_ctr
    delta_err_result = total_delta_err / total_ctr
    scores_pairs, scores_rankings = test_suite.log_final_results(
        additional_metrics=[["rmse", rmse_result], ["mae", mae_result], ["abs_rel", abs_rel_result],
                            ["delta_err_1_25", delta_err_result]])

    logging.info("{}: Results (WKDR, WKDR_eq, WKDR_neq) for {}: {}".format(model_name, dataset_type, scores_pairs))
    logging.info("{}: Results (rankings) for {}: {}".format(model_name, dataset_type, scores_rankings))
    logging.info("{}: RMSE: {}".format(model_name, rmse_result))
    logging.info("{}: MAE: {}".format(model_name, mae_result))
    logging.info("{}: Abs rel: {}".format(model_name, abs_rel_result))
    logging.info("{}: Delta err (1.25): {}".format(model_name, delta_err_result))
    logging.info("Maximum value for dataset {}: {}".format(dataset_type, max_value_so_far))
    logging.info("Mean value for dataset {}: {}".format(dataset_type, mean_value_so_far / total_ctr))


def prepare_data_for_external_run(dataset_type):
    logging.info("Preparing dataset {}...".format(dataset_type))
    config, model_input_shape, eval_config = initialize_evaluation_run()

    dao = get_dao_for_dataset_type(dataset_type, config, model_input_shape)
    dataset = dao.get_test_dataset()

    image_ds = dataset.map(lambda loc_x, loc_y: loc_x)
    image_ds_np = np.array([img for img in TFDatasetTestSuite.tensorflow_dataset_to_numpy(image_ds)])

    output_path = os.path.join(config["DATA"]["CACHE_PATH_PREFIX"], "resized_images_{}.npy".format(dataset_type))
    np.save(output_path, image_ds_np)
    logging.info("Finished preparing dataset {}.".format(dataset_type))


def external_eval(model_name, config, eval_config, model_input_shape, model_type=ModelType.FULLY_FLEDGED_REDWEB,
                  wkdr_threshold=0.03, seed=42):
    for dataset_type in get_datasets_to_be_evaluated(eval_config):
        external_path = os.path.join(config["DATA"]["CACHE_PATH_PREFIX"], "zeroshot_preds/{}_{}_pred_depths.npy".format(
            model_name, dataset_type))

        evaluate_model_predictions(config, model_input_shape, dataset_type, model_name, model_type, None, external_path,
                                   invert_ds_relations=get_inverted_by_dataset_and_model(dataset_type, model_name),
                                   seed=seed, wkdr_threshold=wkdr_threshold)


def internal_eval(model, model_name, model_type, config, eval_config, model_input_shape, wkdr_threshold=0.03, seed=42):
    for dataset_type in get_datasets_to_be_evaluated(eval_config):
        evaluate_model_predictions(config, model_input_shape, dataset_type, model_name, model_type, model, None,
                                   invert_ds_relations=get_inverted_by_dataset_and_model(dataset_type, model_name),
                                   wkdr_threshold=wkdr_threshold, seed=seed)


def load_pl_depth_model(model_path, model_type):
    if model_type == ModelType.FULLY_FLEDGED_REDWEB:
        return keras.models.load_model(model_path,
                                       custom_objects={'HourglassNegativeLogLikelihood': HourglassNegativeLogLikelihood,
                                                       'FeatureFusionLayer': FeatureFusionLayer,
                                                       'AdaptiveOutputLayer': AdaptiveOutputLayer,
                                                       'BottleneckConvLayer': BottleneckConvLayer},
                                       compile=False)
    else:
        custom_objects = {'EffNetFullyFledged': EffNetFullyFledged}

        return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


@click.command()
@click.option('--seed', default=0)
@click.option('--eq_threshold', default=0.0, type=click.FLOAT)
@click.option('--model_name', default='external', type=click.Choice(['external', 'PLDepth', 'MiDaS'],
                                                                    case_sensitive=True))
@click.option('--model_id', type=click.STRING, help='Model ID used to identify the model weight path in eval.ini')
@click.option('--verbose', default=False, type=click.BOOL)
def perform_evaluation(seed, eq_threshold, model_name, model_id, verbose):
    config, model_input_shape, eval_config = initialize_evaluation_run()

    eval_pl_depth = model_name == "PLDepth"
    eval_midas_own_trained = model_name == "MiDaS"

    if eval_pl_depth or eval_midas_own_trained:
        assert model_id != "", "Model ID must be given for PLDepth or self-trained MiDaS model!"

        # Gather model meta information
        model_path = os.path.join(config["DATA"]["CACHE_PATH_PREFIX"], eval_config[model_name][model_id])
        model_type = get_pl_model_type_from_path(model_path)

        if model_type == ModelType.FULLY_FLEDGED_EFFNET:
            model_name = "{}_EffNet".format(model_name)
        elif model_type == ModelType.FULLY_FLEDGED_REDWEB:
            model_name = "{}_ReDWeb".format(model_name)
        else:
            raise NotImplementedError("Unrecognized type '{}' for {} model.".format(model_type, model_name))

        # Load model
        model = load_pl_depth_model(model_path, model_type)
        if verbose:
            model.summary()

        # Execute evaluation
        internal_eval(model, model_name, model_type, config, eval_config, model_input_shape,
                      wkdr_threshold=eq_threshold, seed=seed)

    else:
        # Evaluate baselines on base of externally available predictions
        external_models = eval_config["EXTERNAL"]["models"].split(",")
        for external_model in external_models:
            external_eval(external_model, config, eval_config, model_input_shape, wkdr_threshold=eq_threshold,
                          seed=seed)


@click.command()
@click.option('--dataset', type=click.Choice(['IBIMS', 'SINTEL', 'DIODE', 'TUM', 'HR-WSI']))
def prepare_dataset_for_external_predictions(dataset):
    dataset_type = get_dataset_type_by_name(dataset)

    prepare_data_for_external_run(dataset_type)


if __name__ == "__main__":
    perform_evaluation()
