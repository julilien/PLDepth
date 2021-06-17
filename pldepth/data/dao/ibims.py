import tensorflow as tf
from scipy import io
from skimage.transform import resize
import os
import numpy as np

from pldepth.data.data_meta import TFDataAccessObject, TESTING_ONLY_STR


class IbimsTFDataAccessObject(TFDataAccessObject):
    def __init__(self, root_path, target_shape):
        self.root_path = root_path
        self.target_shape = target_shape

        self.file_names = [s for s in tf.data.Dataset.list_files(os.path.join(self.root_path, '*.mat'),
                                                                 shuffle=False).as_numpy_iterator()]

    def read_raw_mat(self, file_path):
        raw_data = io.loadmat(file_path)['data']
        image = resize(raw_data[0][0][2], self.target_shape, anti_aliasing=True)
        gt = resize(raw_data[0][0][3], self.target_shape[:2], anti_aliasing=True)

        return image.astype(np.float32), gt.astype(np.float32)

    def get_training_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("Ibims", "training"))

    def get_validation_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("Ibims", "validation"))

    def get_test_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.file_names).map(
            lambda loc_x: tf.numpy_function(self.read_raw_mat, inp=[loc_x],
                                            Tout=[tf.float32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
