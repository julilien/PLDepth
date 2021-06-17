import abc
import tensorflow as tf

TESTING_ONLY_STR = "{} is intended to be used for testing only and, thus, this DAO does not provide a {} set."


class TFDatasetDataProvider(abc.ABC):
    """
    Abstract base class abstracting the construction of datasets fed to the model based on TF datasets.
    """

    def __init__(self, model_params):
        self.model_params = model_params

    @abc.abstractmethod
    def provide_train_dataset(self, base_ds, base_ds_gts=None):
        pass

    @abc.abstractmethod
    def provide_val_dataset(self, base_ds, base_ds_gts=None):
        pass


class TFDataAccessObject(abc.ABC):
    @abc.abstractmethod
    def get_training_dataset(self):
        pass

    @abc.abstractmethod
    def get_validation_dataset(self):
        pass

    @abc.abstractmethod
    def get_test_dataset(self):
        pass

    @staticmethod
    def read_file_png(file_path, num_channels=3):
        return tf.cast(tf.image.decode_png(tf.io.read_file(file_path), channels=num_channels), dtype=tf.float32) / 255.

    @staticmethod
    def read_file_jpg(file_path, num_channels=3):
        return tf.cast(tf.image.decode_jpeg(tf.io.read_file(file_path), channels=num_channels), dtype=tf.float32) / 255.
