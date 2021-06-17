from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers

from pldepth.losses.losses_meta import DepthLossType
from pldepth.models.pl_hourglass import FullyFledgedModel


class BottleneckLayer(layers.Layer):
    expansion = 4

    def __init__(self, reg_term, planes, stride=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.planes = planes
        self.stride = stride
        self.downsample = downsample

        if reg_term is not None:
            reg_term = regularizers.l2(reg_term)

        self.conv0 = layers.Conv2D(self.planes, (1, 1), use_bias=False, activity_regularizer=reg_term)
        self.bn0 = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(self.planes, (3, 3), strides=self.stride, use_bias=False, padding="same",
                                   activity_regularizer=reg_term)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(self.planes * 4, (1, 1), use_bias=False, activity_regularizer=reg_term)
        self.bn2 = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        residual = x

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def get_config(self):
        config = {
            'reg_term': self.reg_term,
            'planes': self.planes,
            'stride': self.stride,
            'downsample': layers.serialize(self.downsample),
        }
        base_config = super(BottleneckLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BottleneckConvLayer(layers.Layer):
    def __init__(self, reg_term, in_out_planes=256, conv0=None, bn0=None, conv1=None, bn1=None, conv2=None, bn2=None,
                 conv3=None, bn3=None, conv4=None, bn4=None, conv5=None, bn5=None, relu=None, **kwargs):
        super().__init__(**kwargs)
        self.in_out_planes = in_out_planes
        self.reg_term = reg_term

        if reg_term is not None:
            reg_term = regularizers.l2(reg_term)

        if conv0 is None:
            self.conv0 = layers.Conv2D(self.in_out_planes / 4, (1, 1), use_bias=False,
                                       activity_regularizer=reg_term)
        else:
            self.conv0 = conv0
        if relu is None:
            self.bn0 = layers.BatchNormalization()
        else:
            self.bn0 = bn0

        if conv1 is None:
            self.conv1 = layers.Conv2D(self.in_out_planes / 4, (3, 3), use_bias=False, padding="same",
                                       activity_regularizer=reg_term)
        else:
            self.conv1 = conv1
        if bn1 is None:
            self.bn1 = layers.BatchNormalization()
        else:
            self.bn1 = bn1

        if conv2 is None:
            self.conv2 = layers.Conv2D(self.in_out_planes, (1, 1), use_bias=False, activity_regularizer=reg_term)
        else:
            self.conv2 = conv2
        if bn2 is None:
            self.bn2 = layers.BatchNormalization()
        else:
            self.bn2 = bn2

        if conv3 is None:
            self.conv3 = layers.Conv2D(self.in_out_planes / 4, (1, 1), use_bias=False, activity_regularizer=reg_term)
        else:
            self.conv3 = conv3
        if bn3 is None:
            self.bn3 = layers.BatchNormalization()
        else:
            self.bn3 = bn3

        if conv4 is None:
            self.conv4 = layers.Conv2D(self.in_out_planes / 4, (3, 3), use_bias=False, padding="same",
                                       activity_regularizer=reg_term)
        else:
            self.conv4 = conv4
        if bn4 is None:
            self.bn4 = layers.BatchNormalization()
        else:
            self.bn4 = bn4

        if conv5 is None:
            self.conv5 = layers.Conv2D(self.in_out_planes, (1, 1), use_bias=False, activity_regularizer=reg_term)
        else:
            self.conv5 = conv5
        if bn5 is None:
            self.bn5 = layers.BatchNormalization()
        else:
            self.bn5 = bn5

        if relu is None:
            self.relu = layers.ReLU()
        else:
            self.relu = relu

    def call(self, inputs, **kwargs):
        x = inputs
        residual = x

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        x = self.relu(out)

        residual = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)

        out += residual

        x = self.relu(out)

        return x

    def get_config(self):
        config = {
            'reg_term': self.reg_term,
            'in_out_planes': self.in_out_planes,
        }
        base_config = super(BottleneckConvLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualConv(layers.Layer):
    def __init__(self, reg_term, in_out_planes, **kwargs):
        super().__init__(**kwargs)
        self.in_out_planes = in_out_planes

        self.conv0 = layers.Conv2D(self.in_out_planes, (3, 3), use_bias=False, padding='same',
                                   activity_regularizer=reg_term)
        self.bn0 = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(self.in_out_planes, (3, 3), use_bias=False, padding='same',
                                   activity_regularizer=reg_term)
        self.bn1 = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, inputs, **kwargs):
        residual = inputs

        out = self.relu(inputs)
        out = self.conv0(out)
        out = self.bn0(out)

        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        out += residual
        return out

    def get_config(self):
        config = {
            'reg_term': regularizers.serialize(self.reg_term),
            'in_out_planes': self.in_out_planes,
        }
        base_config = super(layers.Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeatureFusionLayer(layers.Layer):
    def __init__(self, inter_planes, out_planes, reg_term, conv0=None, bn0=None, conv1=None, bn1=None,
                 upsampling=None, **kwargs):
        super().__init__(**kwargs)
        self.inter_planes = inter_planes
        self.out_planes = out_planes
        self.reg_term = reg_term

        if reg_term is not None:
            reg_term = regularizers.l2(reg_term)

        if conv0 is None:
            self.conv0 = layers.Conv2D(self.inter_planes, (3, 3), use_bias=False, padding="same",
                                       activity_regularizer=reg_term)
        else:
            self.conv0 = conv0
        if bn0 is None:
            self.bn0 = layers.BatchNormalization()
        else:
            self.bn0 = bn0

        if conv1 is None:
            self.conv1 = layers.Conv2D(self.inter_planes, (3, 3), use_bias=False, padding="same",
                                       activity_regularizer=reg_term)
        else:
            self.conv1 = conv1
        if bn1 is None:
            self.bn1 = layers.BatchNormalization()
        else:
            self.bn1 = bn1

        if upsampling is None:
            self.upsampling = layers.UpSampling2D(interpolation='bilinear')
        else:
            self.upsampling = upsampling

        self.block_left = BottleneckConvLayer(self.reg_term, self.inter_planes)
        self.block_down = BottleneckConvLayer(self.reg_term, self.out_planes)

    def call(self, inputs, **kwargs):
        in_left = inputs[0]
        in_up = inputs[1]

        x_left = self.conv0(in_left)
        x_left = self.bn0(x_left)
        x_left = self.block_left(x_left)

        x_up = self.conv1(in_up)
        x_up = self.bn1(x_up)

        x = x_left + x_up

        x = self.block_down(x)
        x = self.upsampling(x)

        return x

    def get_config(self):
        config = super(FeatureFusionLayer, self).get_config()

        config.update({
            'inter_planes': self.inter_planes,
            'out_planes': self.out_planes,
            'reg_term': self.reg_term,
        })
        return config


class AdaptiveOutputLayer(layers.Layer):
    def __init__(self, reg_term, conv0=None, bn0=None, relu=None, conv1=None, upsampling=None, conv2=None,
                 leaky_relu=None, **kwargs):
        super().__init__(**kwargs)

        self.reg_term = reg_term

        if reg_term is not None:
            reg_term = regularizers.l2(reg_term)

        if conv0 is None:
            self.conv0 = layers.Conv2D(64, (3, 3), use_bias=True, padding="same", activity_regularizer=reg_term)
        else:
            self.conv0 = conv0
        if bn0 is None:
            self.bn0 = layers.BatchNormalization()
        else:
            self.bn0 = bn0

        if relu is None:
            self.relu = layers.ReLU()
        else:
            self.relu = relu

        if conv1 is None:
            self.conv1 = layers.Conv2D(1, (3, 3), use_bias=True, padding="same", activity_regularizer=reg_term)
        else:
            self.conv1 = conv1
        if upsampling is None:
            self.upsampling = layers.UpSampling2D(interpolation='bilinear')
        else:
            self.upsampling = upsampling

        if conv2 is None:
            self.conv2 = layers.Conv2D(1, (1, 1), padding="same")
        else:
            self.conv2 = conv2
        if leaky_relu is None:
            self.leaky_relu = layers.LeakyReLU(0.1)
        else:
            self.leaky_relu = leaky_relu

    def call(self, inputs, **kwargs):
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.upsampling(x)

        x = self.conv2(x)

        return x

    def get_config(self):
        config = super(AdaptiveOutputLayer, self).get_config()
        config.update({
            'reg_term': self.reg_term,
        })
        return config


class ResNetLayer(layers.Layer):
    def __init__(self, block, add_layers, reg_term, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        self.add_layers = add_layers

        self.in_planes = 64

        self.layer0 = self._make_resnet_layer(32, add_layers[0], reg_term)
        self.layer1 = self._make_resnet_layer(64, add_layers[1], reg_term, stride=2)
        self.layer2 = self._make_resnet_layer(128, add_layers[2], reg_term, stride=2)
        self.layer3 = self._make_resnet_layer(256, add_layers[3], reg_term, stride=2)

        self.conv = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", use_bias=False,
                                  activity_regularizer=reg_term)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.max_pool = layers.MaxPool2D((3, 3), (2, 2), padding="same")

    def _make_resnet_layer(self, planes, blocks, reg_term, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * self.block.expansion:
            downsample = Sequential([
                layers.Conv2D(planes * self.block.expansion, (1, 1), strides=stride, use_bias=False,
                              activity_regularizer=reg_term),
                layers.BatchNormalization()])

        add_layers = [self.block(reg_term, planes, stride, downsample)]
        self.in_planes = planes * self.block.expansion
        for _ in range(1, blocks):
            add_layers.append(self.block(reg_term, planes))

        return Sequential(add_layers)

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x96_96_256 = self.layer0(x)
        x48_48_512 = self.layer1(x96_96_256)
        x24_24_1024 = self.layer2(x48_48_512)
        x12_12_2048 = self.layer3(x24_24_1024)

        return [x96_96_256, x48_48_512, x24_24_1024, x12_12_2048]


class ReDWebNetTFVersion(FullyFledgedModel):
    @staticmethod
    def get_model_and_normalization(input_shape, ranking_size, loss_type=DepthLossType.NLL):
        input_img = layers.Input(input_shape)

        reg_term = None

        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        encoder = ResNet50(include_top=False, input_tensor=input_img)

        for layer in encoder.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        gx112_112_256 = encoder.get_layer("conv2_block3_out").output
        gx56_56_512 = encoder.get_layer("conv3_block4_out").output
        gx_28_28_1024 = encoder.get_layer("conv4_block3_out").output
        gx14_14_2048 = encoder.get_layer("conv5_block3_out").output

        bx28_28_1024 = layers.UpSampling2D(interpolation='bilinear')(gx14_14_2048)

        b48_48 = FeatureFusionLayer(256, 256, reg_term)([gx_28_28_1024, bx28_28_1024])
        b96_96 = FeatureFusionLayer(128, 128, reg_term)([gx56_56_512, b48_48])
        b192_192 = FeatureFusionLayer(64, 64, reg_term)([gx112_112_256, b96_96])
        out = AdaptiveOutputLayer(reg_term)(b192_192)

        output_layer = out

        model = ReDWebNetTFVersion(inputs=input_img, outputs=output_layer)

        return model, preprocess_input
