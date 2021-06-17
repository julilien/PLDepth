import abc
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.applications.efficientnet import EfficientNetB5
from tensorflow.python.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet

from pldepth.losses.losses_meta import DepthLossType


class FullyFledgedModel(tf.keras.Model):
    def __init__(self, asc_depth_order=False, *args, **kwargs):
        """
        Constructing a fully fledged model, i.e., a model that predicts a complete depth map as output.

        :param asc_depth_order: Depth order of the training data (see property documentation for more information)
        :param args: See tf.keras.Model documentation for description.
        :param kwargs: See tf.keras.Model documentation for description.
        """
        super(FullyFledgedModel, self).__init__(*args, **kwargs)
        self.asc_depth_order = asc_depth_order

    @property
    def asc_depth_order(self):
        """
        Indicator whether the depths of elements closer to the camera are lower. E. g., the (pseudo-)depth values in
        the HR-WSI dataset are of descending order, while NYUDv2, Ibims, Sintel and DIODE are of ascending order. This
        property is typically used to check in the ordinal metric calculation whether the relations have to be inverted.

        :return: Returns the property as Boolean value (true for ascending, false for descending order)
        """
        return self._asc_depth_order

    @asc_depth_order.setter
    def asc_depth_order(self, value):
        self._asc_depth_order = value

    @staticmethod
    @abc.abstractmethod
    def get_model_and_normalization(input_shape, ranking_size, loss_type=DepthLossType.NLL):
        pass


class EffNetFullyFledged(FullyFledgedModel):
    @staticmethod
    def get_model_and_normalization(input_shape, ranking_size, loss_type=DepthLossType.NLL):
        input_layer = layers.Input(shape=input_shape, name="input_A")

        encoder = EfficientNetB5(include_top=False, input_tensor=input_layer)

        encoded_layer = encoder.output

        # Freeze encoder layers
        for layer in encoder.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        x = layers.Conv2D(672, (3, 3), padding="same")(encoded_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(interpolation='bilinear')(x)

        # shape 28
        x_add0 = encoder.get_layer("block6a_expand_activation").output
        x = layers.Concatenate()([x, x_add0])

        x = layers.Conv2D(240, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(interpolation='bilinear')(x)

        # shape 56
        x_add1 = encoder.get_layer("block4a_expand_activation").output
        x = layers.Concatenate()([x, x_add1])

        x = layers.Conv2D(144, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(interpolation='bilinear')(x)

        # shape 112
        x_add2 = encoder.get_layer("block3a_expand_activation").output
        x = layers.Concatenate()([x, x_add2])

        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(interpolation='bilinear')(x)

        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D(interpolation='bilinear')(x)

        x = layers.Conv2D(1, (3, 3), padding="same")(x)

        output_layer = x

        return EffNetFullyFledged(inputs=input_layer, outputs=output_layer), preprocess_input_effnet
