import tensorflow as tf
import os
import numpy as np
from skimage.transform import resize

from pldepth.data.data_meta import TFDataAccessObject, TESTING_ONLY_STR


class DIODETFDataAccessObject(TFDataAccessObject):
    def __init__(self, root_path, target_shape):
        self.root_path = root_path
        if len(target_shape) == 3:
            target_shape = target_shape[:2]
        self.target_shape = tf.convert_to_tensor(target_shape)

    def get_training_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("DIODE", "training"))

    def get_validation_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("DIODE", "validation"))

    def get_test_dataset(self):
        file_pattern_images = os.path.join(self.root_path, '*/*/*/*.png')

        # This is necessary to keep the same order of the files
        file_name_images = [s for s in
                            tf.data.Dataset.list_files(file_pattern_images, shuffle=False).as_numpy_iterator()]
        file_name_depths = [s.replace(b'.png', b'_depth.npy') for s in file_name_images]

        def resize_imgs(img):
            return tf.image.resize(img, self.target_shape)

        def read_and_resize_depth(file_path):
            return resize(np.squeeze(np.load(file_path)), self.target_shape[:2])

        images_ds = tf.data.Dataset.from_tensor_slices(file_name_images).map(self.read_file_png,
                                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
            resize_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        gts_ds = tf.data.Dataset.from_tensor_slices(file_name_depths).map(
            lambda loc_x: tf.numpy_function(read_and_resize_depth, [loc_x], Tout=tf.float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return tf.data.Dataset.zip((images_ds, gts_ds))
