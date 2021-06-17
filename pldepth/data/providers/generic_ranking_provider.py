import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
import os

from pldepth.data.data_meta import TFDatasetDataProvider
from pldepth.data.depth_utils import get_depth_relation
from pldepth.util.str_literals import DONE_STR


class GenericHourglassPairRelationDataProvider(TFDatasetDataProvider):
    def __init__(self, model_params, seed, invert_relation_sign, threshold=0.03, cache_val_data=True,
                 save_pairs_on_disk=False, config=None):
        super().__init__(model_params)
        self.seed = seed
        self.invert_relation_sign = invert_relation_sign
        self.threshold = threshold
        self.cache_val_data = cache_val_data
        self.dataset_name = model_params.get_parameter("dataset")
        self.save_pairs_on_disk = save_pairs_on_disk
        if self.save_pairs_on_disk:
            assert config is not None, "If the generated pairs should be saved, a configuration specifying the cache " \
                                       "location must be given!"
        self.config = config

    def provide_train_dataset(self, base_ds, base_ds_gts=None):
        raise NotImplementedError("Training provision is not implemented yet.")

    def provide_val_dataset(self, base_ds, base_ds_gts=None):
        # Set seed
        np.random.seed(self.seed)

        logging.debug("Retrieving validation rankings...")

        cache_path = os.path.join(self.config["DATA"]["CACHE_PATH_PREFIX"], "ordinal_pair_cache/{}_{}_{}_{}.npy".format(
            self.dataset_name, "val", self.model_params.get_parameter("val_rankings_per_img"), self.seed))
        ordinal_pairs = self.retrieve_ordinal_pairs(base_ds, cache_path)
        logging.debug(DONE_STR)

        # Since access_ds is assumed to contain both images and gts, it must be split
        result = tf.data.Dataset.zip((base_ds, tf.data.Dataset.from_tensor_slices(ordinal_pairs)))
        if self.cache_val_data:
            return result.cache()
        else:
            return result

    def provide_test_dataset(self, base_ds):
        # Set seed
        np.random.seed(self.seed)

        logging.debug("Retrieving test rankings...")

        cache_path = os.path.join(self.config["DATA"]["CACHE_PATH_PREFIX"], "ordinal_pair_cache/{}_{}_{}.npy".format(
            self.dataset_name, self.model_params.get_parameter("val_rankings_per_img"), self.seed))
        ordinal_pairs = self.retrieve_ordinal_pairs(base_ds, cache_path)
        logging.debug(DONE_STR)

        # Since access_ds is assumed to contain both images and gts, it must be split
        result = tf.data.Dataset.zip((base_ds, tf.data.Dataset.from_tensor_slices(ordinal_pairs)))
        if self.cache_val_data:
            return result.cache()
        else:
            return result

    def retrieve_ordinal_pairs(self, base_ds, cache_path):
        if not self.save_pairs_on_disk:
            ordinal_pairs = self.generate_ordinal_pairs(base_ds, invert_relation_sign=self.invert_relation_sign)
        elif not os.path.exists(cache_path):
            ordinal_pairs = self.generate_ordinal_pairs(base_ds, invert_relation_sign=self.invert_relation_sign)
            np.save(cache_path, ordinal_pairs)
        else:
            ordinal_pairs = np.load(cache_path)

        logging.debug("Number of unequal relations: {}".format(np.sum(ordinal_pairs[:, :, 2] != 0)))
        logging.debug("Number of equal relations: {}".format(np.sum(ordinal_pairs[:, :, 2] == 0)))

        return ordinal_pairs

    def generate_ordinal_pairs(self, base_ds_imgs_gts, invert_relation_sign=False):
        val_rankings_per_img = self.model_params.get_parameter("val_rankings_per_img")

        result = np.zeros([tf.data.experimental.cardinality(base_ds_imgs_gts), val_rankings_per_img, 5], np.float32)

        with tqdm(total=result.shape[0]) as pbar:

            for idx, elem in enumerate(base_ds_imgs_gts.as_numpy_iterator()):
                image = np.squeeze(elem[0])
                gt = np.squeeze(elem[1])

                for j in range(result.shape[1]):
                    x0 = np.random.randint(image.shape[0])
                    y0 = np.random.randint(image.shape[1])

                    x1 = np.random.randint(image.shape[0])
                    y1 = np.random.randint(image.shape[1])

                    z0 = gt[x0, y0]
                    z1 = gt[x1, y1]

                    point0 = x0 * gt.shape[1] + y0
                    point1 = x1 * gt.shape[1] + y1

                    depth_relation = get_depth_relation(z0, z1, self.threshold)
                    if invert_relation_sign:
                        depth_relation *= -1

                    # Depth values are also stored to be able to compute relations with different thresholds
                    result[idx, j] = np.array([point0, point1, depth_relation, z0, z1])
                pbar.update(1)
        return result


class GenericHourglassRankingDataProvider(TFDatasetDataProvider):
    def __init__(self, model_params, query_ranking_size, seed, invert_relation_sign, threshold=0.03,
                 cache_val_data=True, save_rankings_on_disk=False, config=None):
        super().__init__(model_params)
        self.query_ranking_size = query_ranking_size
        self.seed = seed
        self.invert_relation_sign = invert_relation_sign
        self.threshold = threshold
        self.cache_val_data = cache_val_data
        self.dataset_name = model_params.get_parameter("dataset")
        self.save_rankings_on_disk = save_rankings_on_disk
        if self.save_rankings_on_disk:
            assert config is not None, "If the generated rankings should be saved, a configuration specifying the " \
                                       "cache location must be given!"
        self.config = config

    def provide_train_dataset(self, base_ds, base_ds_gts=None):
        raise NotImplementedError("Providing training data is not supported.")

    def provide_val_dataset(self, base_ds, base_ds_gts=None):
        # Set seed
        np.random.seed(self.seed)

        logging.debug("Generating validation rankings...")

        cache_path = os.path.join(self.config["DATA"]["CACHE_PATH_PREFIX"], "ranking_cache/{}_{}_{}_{}_{}.npy".format(
            self.dataset_name, "val", 100, self.seed, self.query_ranking_size))
        rankings = self.retrieve_rankings(base_ds, cache_path)
        logging.debug(DONE_STR)

        # Since access_ds is assumed to contain both images and gts, it must be split
        result = tf.data.Dataset.zip((base_ds, tf.data.Dataset.from_tensor_slices(rankings)))
        if self.cache_val_data:
            return result.cache()
        else:
            return result

    def provide_test_dataset(self, base_ds):
        # Set seed
        np.random.seed(self.seed)

        logging.debug("Generating test rankings...")

        cache_path = os.path.join(self.config["DATA"]["CACHE_PATH_PREFIX"], "ranking_cache/{}_{}_{}_{}.npy".format(
            self.dataset_name, 100, self.seed, self.query_ranking_size))
        rankings = self.retrieve_rankings(base_ds, cache_path)
        logging.debug(DONE_STR)

        # Since access_ds is assumed to contain both images and gts, it must be split
        result = tf.data.Dataset.zip((base_ds, tf.data.Dataset.from_tensor_slices(rankings)))
        if self.cache_val_data:
            return result.cache()
        else:
            return result

    def retrieve_rankings(self, base_ds, cache_path):
        if not self.save_rankings_on_disk:
            rankings = self.generate_rankings(base_ds, invert_relation_sign=self.invert_relation_sign)
        elif not os.path.exists(cache_path):
            rankings = self.generate_rankings(base_ds, invert_relation_sign=self.invert_relation_sign)
            np.save(cache_path, rankings)
        else:
            rankings = np.load(cache_path)

        return rankings

    def generate_rankings(self, base_ds_imgs_gts, invert_relation_sign=False, val_rankings_per_img=100):
        result = np.zeros([tf.data.experimental.cardinality(base_ds_imgs_gts), val_rankings_per_img,
                           self.query_ranking_size, 2], np.float32)

        with tqdm(total=result.shape[0]) as pbar:
            for idx, elem in enumerate(base_ds_imgs_gts.as_numpy_iterator()):
                gt = np.squeeze(elem[1])

                gt = gt.reshape([-1])

                for ranking_idx in range(result.shape[1]):
                    sampled_points = np.zeros([result.shape[2], 2])
                    sampled_points[0, 0] = np.random.randint(0, len(gt))
                    sampled_points[0, 1] = gt[sampled_points[0, 0].astype(np.int)]

                    for pos_idx in range(1, result.shape[2]):
                        tmp_sample = np.random.randint(0, len(gt))
                        tmp_dist = gt[tmp_sample]
                        sampled_points[pos_idx, 0] = tmp_sample
                        sampled_points[pos_idx, 1] = tmp_dist

                    if invert_relation_sign:
                        # Lower values are closer to the camera => Shall be inverted
                        loc_idxs = np.argsort(sampled_points[:, 1])
                        sampled_points[:, 1] = 1 / (sampled_points[:, 1] + 1)
                        if np.min(sampled_points[:, 1]) < 0.0:
                            logging.warning(
                                "Got a value in the sampling routine that is negative and has been inverted.")

                    else:
                        # Higher values are closer to the camera
                        loc_idxs = np.argsort(sampled_points[:, 1])[::-1]
                    result[idx, ranking_idx] = sampled_points[loc_idxs]

                pbar.update(1)
        return result

    @staticmethod
    def assure_no_equal_relation(distances, curr_depth, position_idx, threshold):
        no_equal_relation = True
        for i in range(position_idx):
            if get_depth_relation(distances[i], curr_depth, threshold=threshold) == 0:
                no_equal_relation = False
        return no_equal_relation
