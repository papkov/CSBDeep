from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
import numpy as np
from scipy import stats

from ..utils import _raise, backend_channels_last

from keras.layers import Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, DepthwiseConv2D
from keras.layers.merge import Concatenate
from keras.models import Sequential
from keras.initializers import glorot_uniform

# TODO deprecate
def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func


# TODO deprecate
def conv_block3(n_filter, n1, n2, n3,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func


def conv_block(n_filter, n1, n2,
               n3=None,
               activation="relu",
               border_mode="same",
               dropout=0.0,
               batch_norm=False,
               init="glorot_uniform",
               name='conv_block_seq',
               seed=None,
               **kwargs):

    # TODO ugly call counting solution, rewrite as a decorator
    conv_block.counter += 1
    n_dim = 2 if n3 is None else 3

    # Select parameters by dimensionality
    conv = Conv2D if n_dim == 2 else Conv3D
    kernel_size = (n1, n2) if n_dim == 2 else (n1, n2, n3)

    # Fill list of layers
    layers = [conv(n_filter, kernel_size, padding=border_mode,
                   kernel_initializer=glorot_uniform(seed=seed) if init == "glorot_uniform" else init,
                   activation=None if batch_norm else activation, **kwargs)]
    if batch_norm:
        layers.append(BatchNormalization())
        layers.append(Activation(activation))
    if dropout is not None and dropout > 0:
        layers.append(Dropout(dropout))

    # Unite layers in Sequential model under the name of Conv layer
    layers = Sequential(layers, name='{:03d}_{}'.format(conv_block.counter, name))

    return layers
conv_block.counter = 0  # Enumerate conv layer


def unet_block(n_depth=2,
               n_filter_base=16,
               kernel_size=(3, 3),
               n_conv_per_depth=2,
               input_planes=1,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               same_seed=False,
               base_seed=0,
               last_activation=None,
               long_skip=True,
               pool=(2, 2),
               prefix=''):

    # Constants
    n_dim = len(kernel_size)
    channel_axis = -1 if backend_channels_last() else 1

    # If sizes do not match, raise errors
    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    if n_dim not in (2, 3):
        raise ValueError('unet_block only 2d or 3d.')

    # Pick appropriate layers
    pooling = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    # Set up activation function
    if last_activation is None:
        last_activation = activation

    def _name(s):
        return prefix+s

    def _func(layer):
        conv_counter = 0
        skip_layers = []

        # down ...
        for n in range(n_depth):

            filters = n_filter_base * 2 ** n

            for i in range(n_conv_per_depth):

                # Calculate last dim of conv input shape
                last_dim = filters
                if i == 0:
                    last_dim = input_planes if n == 0 else filters // 2

                layer = conv_block(filters, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   input_shape=(None,)*n_dim + (last_dim,),
                                   seed=base_seed+conv_counter if same_seed else None,
                                   name=_name("down_level_%s_no_%s" % (n, i)))(layer)
                conv_counter += 1

            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            # Calculate last dim of conv input shape
            last_dim = filters if i == 0 else filters * 2

            layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm,
                               input_shape=(None,)*n_dim + (last_dim,),
                               seed=base_seed+conv_counter if same_seed else None,
                               name=_name("middle_%s" % i))(layer)
        # TODO should it be n_conv_per_depth-1 for name?
        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           batch_norm=batch_norm,
                           input_shape=(None,)*n_dim + (filters * 2,),
                           seed=base_seed+conv_counter if same_seed else None,
                           name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):

            layer = upsampling(pool)(layer)
            # We want to be able to create shortcuts without long skip connections to prevent memorization
            if long_skip:
                layer = Concatenate(axis=channel_axis)([layer, skip_layers[n]])

            filters = n_filter_base * 2 ** n

            for i in range(n_conv_per_depth - 1):

                # Calculate last dim of conv input shape
                last_dim = filters
                if i == 0 and long_skip:
                    last_dim = filters * 2

                layer = conv_block(filters, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   input_shape=(None,)*n_dim + (last_dim,),
                                   seed=base_seed+conv_counter if same_seed else None,
                                   name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm,
                               input_shape=(None,)*n_dim + (filters,),
                               seed=base_seed+conv_counter if same_seed else None,
                               name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func


def unet_blocks(n_blocks=1,
                input_planes=1,
                output_planes=1,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3, 3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                same_seed=True,
                base_seed=0,
                last_activation=None,
                shared_idx=[],
                pool=(2, 2)):

    # TODO write initialization tests

    # Constants
    n_dim = len(kernel_size)
    channel_axis = -1 if backend_channels_last() else 1

    # If sizes do not match, raise errors
    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    if n_dim not in (2, 3):
        raise ValueError('unet_block only 2d or 3d.')

    # Pick appropriate layers
    pooling = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    # Set up activation function
    if last_activation is None:
        last_activation = activation

    # TODO write a wrapper for adding a shared layer
    shared_layers = []

    def _func(layer):
        _func.counter += 1
        conv_counter = 0
        skip_layers = []
        filters = n_filter_base

        # down ...
        for n in range(n_depth):

            filters = n_filter_base * 2 ** n

            for i in range(n_conv_per_depth):

                # Calculate last dim of conv input shape
                last_dim = filters
                if i == 0:
                    last_dim = input_planes if n == 0 else filters // 2

                # print('down input {}, output {}'.format(last_dim, filters))

                # Create conv block (Sequential)
                cb = conv_block(filters, *kernel_size,
                                dropout=dropout,
                                activation=activation,
                                batch_norm=batch_norm,
                                input_shape=(None,) * n_dim + (last_dim,),
                                seed=base_seed+conv_counter if same_seed else None,
                                name="U{}_down_level_{}_no_{}".format(_func.counter, n, i))

                # If we share this conv block, take it from shared layers instead:
                if conv_counter in shared_idx:
                    # We might not find this block, than we need to init it
                    try:
                        cb = shared_layers[conv_counter]
                    except IndexError:
                        shared_layers.append(cb)
                # If we don't share, append None instead to keep indices aligned
                else:
                    shared_layers.append(None)

                layer = cb(layer)
                conv_counter += 1

            skip_layers.append(layer)
            layer = pooling(pool, name="U{}_max_{}".format(_func.counter, n))(layer)

        # middle
        for i in range(n_conv_per_depth):
            # Calculate last dim of conv input shape
            last_dim = filters
            filters = n_filter_base * 2 ** (n_depth if i != (n_conv_per_depth - 1) else max(0, n_depth - 1))

            # print('middle input {}, output {}'.format(last_dim, filters))

            cb = conv_block(filters,
                            *kernel_size,
                            dropout=dropout,
                            activation=activation,
                            batch_norm=batch_norm,
                            input_shape=(None,) * n_dim + (last_dim,),
                            seed=base_seed+conv_counter if same_seed else None,
                            name="U{}_middle_{}".format(_func.counter, i))

            # If we share this conv block, take it from shared layers instead:
            if conv_counter in shared_idx:
                # We might not find this block, than we need to init it
                try:
                    cb = shared_layers[conv_counter]
                except IndexError:
                    shared_layers.append(cb)
            # If we don't share, append None instead to keep indices aligned
            else:
                shared_layers.append(None)

            layer = cb(layer)
            conv_counter += 1

        # ...and up with skip layers
        for n in reversed(range(n_depth)):

            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            filters = n_filter_base * 2 ** n

            for i in range(n_conv_per_depth):

                # Calculate last dim of conv input shape
                last_dim = filters
                if i == 0:
                    last_dim = filters * 2

                filters = filters if i < n_conv_per_depth - 1 else n_filter_base * 2 ** max(0, n - 1)

                # print('up input {}, output {}'.format(last_dim, filters))

                cb = conv_block(filters, *kernel_size,
                                dropout=dropout,
                                # TODO remove last_activation here?
                                activation=activation if (n > 0) and (i == n_conv_per_depth - 1) else last_activation,
                                batch_norm=batch_norm,
                                input_shape=(None,) * n_dim + (last_dim,),
                                seed=base_seed+conv_counter if same_seed else None,
                                name="U{}_up_level_{}_no_{}".format(_func.counter, n, i))

                # If we share this conv block, take it from shared layers instead:
                if conv_counter in shared_idx:
                    # We might not find this block, than we need to init it
                    try:
                        cb = shared_layers[conv_counter]
                    except IndexError:
                        shared_layers.append(cb)
                # If we don't share, append None instead to keep indices aligned
                else:
                    shared_layers.append(None)

                layer = cb(layer)
                conv_counter += 1

        # Combine output to produce output_planes = input_planes
        cb = conv_block(output_planes, 1, 1,
                        n3=1 if n_dim == 3 else None,
                        activation=activation,
                        batch_norm=batch_norm,
                        input_shape=(None,) * n_dim + (filters,),
                        seed=base_seed+conv_counter if same_seed else None,
                        name="U{}_last_conv".format(_func.counter))
        if conv_counter in shared_idx:
            # We might not find this block, than we need to init it
            try:
                cb = shared_layers[conv_counter]
            except IndexError:
                shared_layers.append(cb)
        # If we don't share, append None instead to keep indices aligned
        else:
            shared_layers.append(None)

        layer = cb(layer)
        return layer
    _func.counter = 0

    blocks = [_func for _ in range(n_blocks)]
    #
    #
    # blocks = [unet_block(n_depth=n_depth,
    #                      n_filter_base=n_filter_base,
    #                      kernel_size=kernel_size,
    #                      input_planes=n_blocks-1,
    #                      n_conv_per_depth=n_conv_per_depth,
    #                      activation=activation,
    #                      batch_norm=batch_norm,
    #                      dropout=dropout,
    #                      last_activation=last_activation,
    #                      pool=pool,
    #                      prefix='{}U{}_'.format(prefix, i))
    #           for i in range(n_blocks)]

    return blocks


def gaussian_2d(in_channels=1, k=7, s=3):
    """
    Returns a DepthwiseConv2D non-trainable Gaussian layer.
    in_channels, int: number of input channels
    k, int: kernel size
    s, int: sigma

    Source: https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras
    """

    def _kernel():
        x = np.linspace(-s, s, k+1)
        kernel = np.diff(stats.norm.cdf(x))
        kernel = np.outer(kernel, kernel)
        kernel /= kernel.sum()

        # we need to modify it to make it compatible with the number of input channels
        kernel = np.expand_dims(kernel, axis=-1)
        kernel = np.repeat(kernel, in_channels, axis=-1)  # apply the same filter on all the input channels
        kernel = np.expand_dims(kernel, axis=-1)  # for shape compatibility reasons

        return kernel

    def _layer(inp):
        gaussian_layer = DepthwiseConv2D(k, use_bias=False, padding='same', name='gaussian_blur_block')
        output = gaussian_layer(inp)
        # print(weights.shape, gaussian_layer.get_weights()[0].shape)
        gaussian_layer.set_weights([_kernel()])
        gaussian_layer.trainable = False

        return output

    return _layer



