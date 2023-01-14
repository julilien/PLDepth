import abc
import mlflow
import tensorflow_datasets as tfds
import numpy as np

from pldepth.data.io_utils import Dataset
from pldepth.eval.eval_utils import get_depth_cap_by_dataset


class FullyFledgedTestSuite(abc.ABC):
    def __init__(self, test_images, test_gts, pair_evaluation_metrics, ranking_evaluation_metrics, dataset_type,
                 model_name, seed, query_ranking_size, wkdr_threshold=0.03, invert_rankings_again=False):
        super().__init__()

        self.test_images = test_images
        self.test_gts = test_gts
        self.ranking_gts = None

        assert pair_evaluation_metrics is not None, "You must give valid metrics for the final result calculation!"

        self.pair_evaluation_metrics = pair_evaluation_metrics
        self.ranking_evaluation_metrics = ranking_evaluation_metrics

        self.dataset_type = dataset_type
        self.model_name = model_name
        self.seed = seed
        self.wkdr_threshold = wkdr_threshold
        self.query_ranking_size = query_ranking_size
        self.invert_rankings_again = invert_rankings_again

    @abc.abstractmethod
    def provide_test_image_iterator(self):
        pass

    def submit_prediction(self, image_index, prediction):
        assert 0 <= image_index <= len(self.test_gts), "The given image index must be a valid index given the GT data."

        # Prediction is a full map. Thus, we have to retrieve the relations from the gts
        prediction = np.squeeze(prediction).reshape([-1])
        image_relations = self.test_gts[image_index]
        pred_idxs = image_relations[:, :2].astype(np.int)
        preds = prediction[pred_idxs]
        y_true = image_relations[:, 3:]

        if self.dataset_type == Dataset.DIODE:
            y_true = np.where(np.equal(y_true, 0.), get_depth_cap_by_dataset(self.dataset_type), y_true)
            y_true = np.reshape(y_true, [-1, 2])

        for eval_metric in self.pair_evaluation_metrics:
            eval_metric.update_state(y_true, np.reshape(preds, [-1, 2]))

        # Ranking metrics
        if self.ranking_gts is not None:
            rankings = self.ranking_gts[image_index]
            pred_idxs = np.squeeze(rankings[:, :, 0]).astype(np.int)

            # Prediction is given as predicted value per position with closer to camera being higher
            preds = prediction[pred_idxs].reshape([-1, rankings.shape[1]])
            if self.invert_rankings_again:
                preds = 1. / (preds + 1e-5)

            # Performing argsort twice delivers a ranking
            preds = preds.argsort().argsort()

            # Since elements are sorted s. t. closer elements have higher (better) rank, they need to have higher
            # relevance, too
            y_true = np.reshape(rankings[:, :, 1], [-1, rankings.shape[1]])

            if self.dataset_type == Dataset.SINTEL:
                # Sintel is already inverted. Thus, we consider their values in [0, 1]
                y_true = y_true / get_depth_cap_by_dataset(self.dataset_type)

            for eval_metric in self.ranking_evaluation_metrics:
                eval_metric.update_state(y_true, np.reshape(preds, [-1, self.query_ranking_size]))

    def submit_batch_predictions(self, image_mask, predictions):
        predictions = np.squeeze(predictions).reshape([predictions.shape[0], -1])

        image_relations = self.test_gts[image_mask]
        pred_idxs = image_relations[:, :, :2].astype(np.int)

        preds = np.take(predictions, pred_idxs)

        y_true = image_relations[:, :, 3:]

        for eval_metric in self.pair_evaluation_metrics:
            # eval_metric.update_state(np.reshape(image_relations[:, 2], [-1]), np.reshape(preds, [-1, 2]))
            eval_metric.update_state(np.reshape(y_true, [-1, 2]), np.reshape(preds, [-1, 2]))

    def log_final_results(self, additional_metrics=None):
        scores_pairs = [metric.result().numpy() for metric in self.pair_evaluation_metrics]
        scores_rankings = [metric.result().numpy() for metric in self.ranking_evaluation_metrics]

        with mlflow.start_run():
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("dataset", self.dataset_type)
            mlflow.log_param("pairs_per_image", self.test_gts.shape[1])
            mlflow.log_param("seed", self.seed)
            mlflow.log_param("wkdr_threshold", self.wkdr_threshold)

            for score_idx, score in enumerate(scores_pairs):
                mlflow.log_metric(self.pair_evaluation_metrics[score_idx].name, score)

            ranking_metric_names = ["dcg", "ndcg"]
            for score_idx, score in enumerate(scores_rankings):
                mlflow.log_metric(ranking_metric_names[score_idx], score)

            if additional_metrics is not None:
                for add_metric in additional_metrics:
                    mlflow.log_metric(add_metric[0], add_metric[1])

        for metric in self.pair_evaluation_metrics:
            metric.reset_states()

        for metric in self.ranking_evaluation_metrics:
            metric.reset_states()

        return scores_pairs, scores_rankings


class SimpleTestSuite(FullyFledgedTestSuite):
    def provide_test_image_iterator(self):
        return self.test_images


class TFDatasetTestSuite(FullyFledgedTestSuite):
    def __init__(self, pair_evaluation_metrics, ranking_evaluation_metrics, dataset_type, model_name, pair_dataset,
                 ranking_dataset, seed, query_ranking_size, wkdr_threshold=0.03, invert_rankings_again=False):
        super().__init__(None, None, pair_evaluation_metrics, ranking_evaluation_metrics, dataset_type, model_name,
                         seed, query_ranking_size, wkdr_threshold, invert_rankings_again)
        pair_dataset_np = TFDatasetTestSuite.tensorflow_dataset_to_numpy(pair_dataset)

        self.test_gts = []
        for elem in pair_dataset_np:
            self.test_gts.append(np.squeeze(elem[1]))
        self.test_gts = np.array(self.test_gts)

        if ranking_dataset is not None:
            ranking_dataset_np = TFDatasetTestSuite.tensorflow_dataset_to_numpy(ranking_dataset)

            self.ranking_gts = []
            for elem in ranking_dataset_np:
                self.ranking_gts.append(elem[1])
            self.ranking_gts = np.array(self.ranking_gts)
        else:
            self.ranking_gts = None

        self.test_images = pair_dataset

    @staticmethod
    def tensorflow_dataset_to_numpy(dataset):
        return tfds.as_numpy(dataset)

    def provide_test_image_iterator(self):
        return self.test_images.as_numpy_iterator()
