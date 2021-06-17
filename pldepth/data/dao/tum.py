import os
import numpy as np
from skimage.transform import resize

import h5py
import tensorflow as tf

from pldepth.data.data_meta import TFDataAccessObject, TESTING_ONLY_STR


class TUMTFDataAccessObject(TFDataAccessObject):
    def __init__(self, root_path, target_shape):
        self.root_path = root_path
        if len(target_shape) == 3:
            target_shape = target_shape[:2]
        self.target_shape = target_shape

    def get_training_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("TUM", "training"))

    def get_validation_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("TUM", "validation"))

    def get_test_dataset(self):
        file_pattern_images = os.path.join(self.root_path, '*.h5')

        def read_h5(file_path):
            with h5py.File(file_path.decode("utf-8"), 'r') as h5_file:
                image = np.array(h5_file['gt']['img_1'], np.float32)
                # Use flow instead of ground truth (cf. supplementary material)
                gt = np.array(h5_file['gt']['pp_depth'], np.float32)

            resized_gt = resize(gt, self.target_shape)

            return resize(image, self.target_shape), resized_gt

        dataset = tf.data.Dataset.list_files(file_pattern_images, shuffle=False).map(
            lambda loc_x: tf.numpy_function(read_h5, [loc_x], Tout=[tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset
