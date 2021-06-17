from tensorflow.python.ops.image_ops_impl import ResizeMethod

from pldepth.data.data_meta import TFDataAccessObject
import os
import tensorflow as tf


class HRWSITFDataAccessObject(TFDataAccessObject):
    def __init__(self, root_path, target_shape, seed):
        self.root_path = root_path
        if len(target_shape) == 3:
            self.target_shape = target_shape[:2]
        self.seed = seed

    def get_training_dataset(self):
        return self.construct_raw_file_dataset('train', zip_ds=False, shuffle=True)

    def get_validation_dataset(self):
        return self.construct_raw_file_dataset('val', zip_ds=False, shuffle=False)

    def get_test_dataset(self, zip_ds=True, exclude_mask=True):
        result_ds = self.construct_raw_file_dataset('val', zip_ds=zip_ds, shuffle=False)
        if exclude_mask:
            result_ds = result_ds.map(lambda loc_x, loc_y, loc_z: (loc_x, loc_y),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return result_ds

    def get_combined_dataset(self):
        return self.construct_raw_file_dataset('*', zip_ds=False, shuffle=True)

    def get_file_dataset(self, file_names, file_extension=".jpg"):
        file_ds = tf.data.Dataset.from_tensor_slices(file_names)

        def png_fn(file):
            return self.read_file_png(file, num_channels=1)

        if file_extension == ".jpg":
            file_ds = file_ds.map(self.read_file_jpg, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif file_extension == ".png":
            file_ds = file_ds.map(png_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            raise NotImplementedError("Unsupported file extension '{}'.".format(file_extension))

        return file_ds

    def construct_raw_file_dataset(self, set_indicator, zip_ds=True, shuffle=False):
        # Synchronize file names for correct order
        file_names_imgs = [s for s in tf.data.Dataset.list_files(os.path.join(self.root_path,
                                                                              '{}/{}/*{}'.format(set_indicator,
                                                                                                 "imgs", ".jpg")),
                                                                 shuffle=shuffle, seed=self.seed).as_numpy_iterator()]
        file_names_gts = [s.replace(b'imgs', b'gts').replace(b'.jpg', b'.png') for s in file_names_imgs]
        file_names_masks = [s.replace(b'imgs', b'valid_masks').replace(b'.jpg', b'.png') for s in
                            file_names_imgs]

        def resize_imgs(img):
            return tf.image.resize(img, self.target_shape)

        image_files_ds = self.get_file_dataset(file_names_imgs, ".jpg").map(resize_imgs,
                                                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_files_ds = self.get_file_dataset(file_names_gts, ".png").map(resize_imgs,
                                                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def resize_masks(mask):
            return tf.squeeze(tf.image.resize(mask, self.target_shape, ResizeMethod.NEAREST_NEIGHBOR))

        cons_masks_ds = self.get_file_dataset(file_names_masks,
                                              ".png").map(resize_masks,
                                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if zip_ds:
            return tf.data.Dataset.zip((image_files_ds, labels_files_ds, cons_masks_ds))
        else:
            return image_files_ds, labels_files_ds, cons_masks_ds
