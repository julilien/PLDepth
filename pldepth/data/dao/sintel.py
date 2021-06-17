import tensorflow as tf
import os

from pldepth.data.data_meta import TFDataAccessObject, TESTING_ONLY_STR


class SintelTFDataAccessObject(TFDataAccessObject):
    def __init__(self, root_path, target_shape):
        self.root_path = root_path
        if len(target_shape) == 3:
            target_shape = target_shape[:2]
        self.target_shape = tf.convert_to_tensor(target_shape)

    def get_training_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("Sintel", "training"))

    def get_validation_dataset(self):
        raise NotImplementedError(TESTING_ONLY_STR.format("Sintel", "validation"))

    def get_test_dataset(self):
        file_pattern_images = os.path.join(self.root_path, 'images/*/*.png')

        # This is necessary to keep the same order of the files
        file_names_imgs = [s for s in
                           tf.data.Dataset.list_files(file_pattern_images, shuffle=False).as_numpy_iterator()]
        file_names_gts = [s.replace(b'/images/', b'/depth_viz/') for s in file_names_imgs]

        def png_fn(file):
            return self.read_file_png(file, num_channels=1) * 255.

        def resize_imgs(img):
            return tf.image.resize(img, self.target_shape)

        def resize_depth(gt):
            return tf.squeeze(tf.image.resize(gt, self.target_shape))

        images_ds = tf.data.Dataset.from_tensor_slices(file_names_imgs) \
            .map(self.read_file_png, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
            resize_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        gts_ds = tf.data.Dataset.from_tensor_slices(file_names_gts) \
            .map(png_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
            resize_depth, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return tf.data.Dataset.zip((images_ds, gts_ds))
