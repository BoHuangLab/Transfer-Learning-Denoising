import tensorflow as tf
from tensorflow import keras as ks
import numpy as np


def build_3d_conv_unet(inputs, use_bias=True):
    """
    build a regular 2D, single frame u-net
    :param inputs:
    :return:
    """
    def flatten_time(layer, padding_mode = "same", use_bias=True):
        # create a convolution with the same size as the number of time points to reduce to one
        #same number of channels as original
        n_time = layer.shape[1].value
        num_channels = layer.shape[-1].value
        layer = ks.layers.Conv3D(filters=num_channels,padding=padding_mode,kernel_size=[n_time, 1, 1], activation='relu', use_bias=use_bias)(layer)
        layer = ks.layers.Lambda(lambda x: x[:, 0, :, :, :])(layer)
        return layer

    def u_net_down_block(layer, index, max_pool=True,padding_mode = "same", use_bias=True):
        #same as regular u-net, but add in 1x1x3 convlutions to time axis
        num_channels = 16 * 2 ** index
        if max_pool:
            layer = ks.layers.TimeDistributed(ks.layers.MaxPool2D((2, 2)))(layer)
        layer = ks.layers.Conv3D(filters=num_channels, padding=padding_mode, kernel_size=[3, 3, 3], activation='relu', use_bias=use_bias)(layer)
        layer = ks.layers.Conv3D(filters=num_channels, padding=padding_mode, kernel_size=[1, 3, 3], activation='relu', use_bias=use_bias)(layer)
        return layer

    def u_net_up_block(layer, concat_tensor, padding_mode = "same", use_bias=True):
        num_channels = layer.shape[3].value // 2
        layer = ks.layers.Conv2DTranspose(filters=num_channels, kernel_size=2, strides=(2, 2), padding=padding_mode, use_bias=use_bias)(layer)
        # concatenate with skip connection
        #offset = (concat_tensor.shape[2].value - layer.shape[2].value) / 2
        #cropped = ks.layers.Lambda(lambda x: x[:, :, int(np.floor(offset)):-int(np.ceil(offset)),
        #                                    int(np.floor(offset)):-int(np.ceil(offset)), :])(concat_tensor)
        cropped = concat_tensor
        # time only convolution to reduce to 2D
        time_flat_cropped = flatten_time(cropped)

        layer = ks.layers.Concatenate(axis=3)([time_flat_cropped, layer])
        layer = ks.layers.Conv2D(filters=num_channels, kernel_size=3, padding=padding_mode, activation='relu', use_bias=use_bias)(layer)
        layer = ks.layers.Conv2D(filters=num_channels, kernel_size=3, padding=padding_mode, activation='relu', use_bias=use_bias)(layer)
        return layer

    padding_mode = "same"
    # downsampling path
    conv0_1 = u_net_down_block(inputs, index=0, max_pool=False, padding_mode = padding_mode, use_bias=use_bias)
    conv1_1 = u_net_down_block(conv0_1, index=1, padding_mode = padding_mode, use_bias=use_bias)
    conv2_1 = u_net_down_block(conv1_1, index=2, padding_mode = padding_mode, use_bias=use_bias)
    conv3_1 = u_net_down_block(conv2_1, index=3, padding_mode = padding_mode, use_bias=use_bias)
    conv4_1 = u_net_down_block(conv3_1, index=4, padding_mode = padding_mode, use_bias=use_bias)

    time_flat = flatten_time(conv4_1, padding_mode = padding_mode, use_bias=use_bias)

    # upsampling path
    conv5_1 = u_net_up_block(time_flat, concat_tensor=conv3_1, padding_mode = padding_mode, use_bias=use_bias)
    conv6_1 = u_net_up_block(conv5_1, concat_tensor=conv2_1,padding_mode = padding_mode, use_bias=use_bias)
    conv7_1 = u_net_up_block(conv6_1, concat_tensor=conv1_1,padding_mode = padding_mode, use_bias=use_bias)
    conv8_1 = u_net_up_block(conv7_1, concat_tensor=conv0_1,padding_mode = padding_mode, use_bias=use_bias)
    outputs = ks.layers.Conv2D(filters=1, kernel_size=1, padding = padding_mode, activation=None, use_bias=use_bias)(conv8_1)

    model = ks.models.Model(inputs=inputs, outputs=outputs)
    return model


def build_model(image_size, n_time_points, type, use_bias=True):
    if type == 'time CNN':
        inputs = ks.layers.Input(shape=(n_time_points, *image_size, 1))
        return build_3d_conv_unet(inputs, use_bias=use_bias)




