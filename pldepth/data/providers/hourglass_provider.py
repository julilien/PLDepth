import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm

from pldepth.data.data_meta import TFDatasetDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy
from pldepth.losses.losses_meta import DepthLossType
import itertools

from pldepth.util.str_literals import DONE_STR


class HourglassLargeScaleDataProvider(TFDatasetDataProvider):
    def __init__(self, model_params, train_consistency_masks, val_consistency_masks, loss_type=DepthLossType.NLL,
                 augmentation=False, sampling_eq_threshold=0.03):
        super().__init__(model_params)
        self.train_consistency_masks = train_consistency_masks
        self.val_consistency_masks = val_consistency_masks

        self.random_sampler = ThresholdedMaskedRandomSamplingStrategy(model_params, sampling_eq_threshold)

        self.augmentation = augmentation

        self.loss_type = loss_type

    def provide_train_dataset(self, base_ds, base_ds_gts=None):
        shuffle_buffer_size = 1024

        imgs_gts_ds = tf.data.Dataset.zip((base_ds, self.train_consistency_masks, base_ds_gts))

        if self.augmentation:
            def augment_fn(loc_img, loc_mask, loc_gt):
                do_flip = tf.random.uniform([]) > 0.5

                loc_img = tf.cond(do_flip, lambda: tf.image.flip_left_right(loc_img), lambda: loc_img)

                loc_mask = tf.cond(do_flip,
                                   lambda: tf.image.flip_left_right(tf.expand_dims(tf.squeeze(loc_mask), axis=-1)),
                                   lambda: loc_mask)
                loc_gt = tf.cond(do_flip, lambda: tf.image.flip_left_right(tf.expand_dims(tf.squeeze(loc_gt), axis=-1)),
                                 lambda: loc_gt)

                loc_mask = tf.squeeze(loc_mask)
                loc_gt = tf.squeeze(loc_gt)

                return loc_img, loc_mask, loc_gt

            imgs_gts_ds = imgs_gts_ds.map(augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        imgs_gts_ds = imgs_gts_ds.shuffle(shuffle_buffer_size)

        ranking_ds = imgs_gts_ds.map(lambda loc_x, loc_y, loc_z: tf.numpy_function(self.sample_rankings,
                                                                                   [loc_x, loc_y, loc_z],
                                                                                   [tf.float32, tf.float32]),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ranking_ds.batch(
            self.model_params.get_parameter("batch_size"), drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE).repeat()

    def provide_val_dataset(self, base_ds, base_ds_gts=None):
        imgs_gts_ds = tf.data.Dataset.zip((base_ds, self.val_consistency_masks, base_ds_gts))
        # Generates validation rankings in advance to keep them the same
        logging.debug("Generating validation rankings...")
        val_rankings = self.generate_validation_rankings(imgs_gts_ds)
        logging.debug(DONE_STR)

        val_rankings_ds = tf.data.Dataset.from_tensor_slices(val_rankings)
        return tf.data.Dataset.zip((base_ds, val_rankings_ds)).batch(
            self.model_params.get_parameter("batch_size"), drop_remainder=True).cache()

    def sample_rankings(self, image, cons_mask, gt, sampling_strategy=None, rankings_per_img=None, return_image=True):
        if sampling_strategy is None:
            sampling_strategy = self.model_params.get_parameter("sampling_strategy")
        if rankings_per_img is None:
            rankings_per_img = self.model_params.get_parameter("rankings_per_image")

        result = sampling_strategy.sample_masked_point_batch(image, cons_mask, gt, rankings_per_img,
                                                             batch_size_factor=5)

        if not return_image:
            return result.astype(np.float32)
        return image.astype(np.float32), result.astype(np.float32)

    @staticmethod
    def mask_gt_merge_fn(img, mask, gt):
        mask = tf.where(tf.greater(mask, 0.), tf.ones_like(mask), tf.zeros_like(mask))

        return img, tf.concat(
            [tf.expand_dims(tf.squeeze(gt), axis=-1), tf.expand_dims(tf.squeeze(mask), axis=-1)], axis=-1)

    @staticmethod
    def construct_batch_combination_matrix(batch_segments):
        result = []
        for i in range(batch_segments.shape[0]):
            segments = batch_segments[i]
            result_cs = HourglassLargeScaleDataProvider.construct_combination_matrix_np(segments)
            result.append(result_cs)
        return result

    @staticmethod
    def construct_combination_matrix_np(segments):
        n_unique = np.unique(segments[:, 1])

        result_cs = []
        for k in range(len(n_unique)):
            segments_mask = segments[:, 1] >= n_unique[k]
            rem_segment_sizes = int(np.sum(segments_mask))

            combs = [i for i in itertools.product([0, 1], repeat=rem_segment_sizes)]

            for c in combs:
                if sum(c) == 0:
                    combs.remove(c)

            tmp_c = np.zeros([len(combs), segments.shape[0]])
            for row_idx, c in enumerate(combs):
                tmp_c[row_idx][segments_mask] = c
            result_cs.append(tmp_c.tolist())
        return result_cs

    @staticmethod
    def construct_combination_matrix(input_rankings, rankings_per_img):
        def batch_fn(batch_dim):
            def instance_fn(batch_segment):
                ranking_segments = tf.cast(batch_segment[:, 1], dtype=tf.int32)
                ranking_segments_counts = tf.math.segment_sum(tf.ones_like(ranking_segments), ranking_segments)
                segment_range = tf.range(tf.constant(0, dtype=tf.int32),
                                         tf.cast(tf.shape(ranking_segments_counts)[0], dtype=tf.int32), dtype=tf.int32)

                def inner_fn(range_ids):
                    def inner_inner_fn(ranking_segment_counts, range_id):
                        n_skip_elements = tf.reduce_sum(ranking_segment_counts[:range_id])
                        repeat_elements = tf.reduce_sum(ranking_segment_counts[range_id:], keepdims=True)

                        def construct_mesh(num_repeats):
                            a = tf.tile(tf.expand_dims(tf.constant([0, 1]), axis=-1), [1, num_repeats[0]])

                            arguments = tf.unstack(a, axis=1)
                            mesh_grid = tf.meshgrid(*arguments, indexing='ij')
                            return tf.reshape(tf.stack(mesh_grid, axis=-1), (-1, num_repeats[0]))[1:]

                        combs = construct_mesh(repeat_elements)
                        return tf.cond(tf.equal(n_skip_elements, 0), lambda: combs, lambda: tf.concat(
                            [tf.zeros([combs.shape[0], n_skip_elements], dtype=combs.dtype), combs], axis=1))

                    def py_fun_mesh(loc_x, loc_y):
                        return tf.py_function(inner_inner_fn, [loc_x, loc_y], tf.int32)

                    inner_result = py_fun_mesh(ranking_segments_counts, range_ids)

                    return tf.RaggedTensor.from_tensor(inner_result)

                return tf.map_fn(inner_fn, segment_range, fn_output_signature=tf.RaggedTensorSpec(shape=[None, None],
                                                                                                  dtype=tf.int32))

            return tf.map_fn(instance_fn, batch_dim, fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, None],
                                                                                             dtype=tf.int32))

        return tf.map_fn(batch_fn, input_rankings,
                         fn_output_signature=tf.RaggedTensorSpec(shape=[rankings_per_img, None, None, None],
                                                                 dtype=tf.int32))

    def sample_partial_rankings(self, image, cons_mask, gt, rankings_per_img=None,
                                return_image=True):
        sampling_strategy = self.model_params.get_parameter("sampling_strategy")
        if rankings_per_img is None:
            rankings_per_img = self.model_params.get_parameter("rankings_per_image")

        result = sampling_strategy.sample_masked_point_batch(image, cons_mask, gt, rankings_per_img)

        if not return_image:
            return result.astype(np.float32)
        return image.astype(np.float32), result.astype(np.float32)

    def generate_validation_rankings(self, access_ds_imgs_gts):
        val_rankings_per_img = self.model_params.get_parameter("val_rankings_per_img")
        ranking_size = self.model_params.get_parameter("ranking_size")

        result = np.zeros([tf.data.experimental.cardinality(access_ds_imgs_gts), val_rankings_per_img, ranking_size, 2],
                          np.float32)
        with tqdm(total=result.shape[0]) as pbar:
            for i, elem in enumerate(access_ds_imgs_gts.as_numpy_iterator()):
                image = elem[0]
                mask = elem[1]
                gt = elem[2]
                result[i] = self.sample_rankings(image, mask, gt, self.random_sampler, val_rankings_per_img,
                                                 return_image=False)
                pbar.update(1)
        return result
