from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last

from keras.layers import Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D
from keras.layers.merge import Concatenate
from keras.models import Sequential


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
               **kwargs):

    # TODO ugly call counting solution, rewrite as a decorator
    conv_block.counter += 1
    n_dim = 2 if n3 is None else 3

    # Select parameters by dimensionality
    conv = Conv2D if n_dim == 2 else Conv3D()
    kernel_size = (n1, n2) if n_dim == 2 else (n1, n2, n3)

    # Fill list of layers
    layers = [conv(n_filter, kernel_size, padding=border_mode, kernel_initializer=init,
                   activation=None if batch_norm else activation, **kwargs)]
    if batch_norm:
        layers.append(BatchNormalization())
        layers.append(Activation(activation))
    if dropout is not None and dropout > 0:
        layers.append(Dropout(dropout))

    # Unite layers in Sequential model under the name of Conv layer
    layers = Sequential(layers, name='{:02d}_{}'.format(conv_block.counter, name))

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
               last_activation=None,
               shared_middle=None,
               pool=(2, 2),
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2, 3):
        raise ValueError('unet_block only 2d or 3d.')

    pooling = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(layer):
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
                                   name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        if shared_middle is None:
            for i in range(n_conv_per_depth - 1):
                # Calculate last dim of conv input shape
                last_dim = filters if i == 0 else filters * 2

                layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   input_shape=(None,)*n_dim + (last_dim,),
                                   name=_name("middle_%s" % i))(layer)
            # TODO should it be n_conv_per_depth-1 for name?
            layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm,
                               input_shape=(None,)*n_dim + (filters * 2,),
                               name=_name("middle_%s" % n_conv_per_depth))(layer)
        else:
            layer = shared_middle(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):

            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            filters = n_filter_base * 2 ** n

            for i in range(n_conv_per_depth - 1):

                # Calculate last dim of conv input shape
                last_dim = filters
                if i == 0:
                    last_dim = filters * 2

                layer = conv_block(filters, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   input_shape=(None,)*n_dim + (last_dim,),
                                   name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm,
                               input_shape=(None,)*n_dim + (filters,),
                               name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func


def unet_blocks(n_blocks=1,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3, 3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                last_activation=None,
                share_middle=False,
                pool=(2, 2),
                prefix=''):

    n_dim = len(kernel_size)

    # Shared layers default
    downsample = None
    middle = None
    upsample = None

    # Shared middle layer
    if share_middle:
        filters = n_filter_base * 2 ** n_depth
        middle = Sequential(name='middle_filters_{}'.format('_'.join([filters] * (n_conv_per_depth - 1) + [filters // 2])))

        for i in range(n_conv_per_depth - 1):
            last_dim = n_filter_base * 2 ** max(0, n_depth - 1) if i == 0 else filters
            middle.add(conv_block(filters, *kernel_size,
                                  activation=activation,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  input_shape=(None,)*n_dim + (last_dim,),
                                  name='middle_{}'.format(i)))

        middle.add(conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                              activation=activation,
                              batch_norm=batch_norm,
                              dropout=dropout,
                              input_shape=(None,)*n_dim + (filters,),
                              name='middle_{}'.format(n_conv_per_depth - 1)))

    blocks = [unet_block(n_depth=n_depth,
                         n_filter_base=n_filter_base,
                         kernel_size=kernel_size,
                         input_planes=n_blocks-1,
                         n_conv_per_depth=n_conv_per_depth,
                         activation=activation,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         last_activation=last_activation,
                         shared_middle=middle,
                         pool=pool,
                         prefix='{}U{}_'.format(prefix, i))
              for i in range(n_blocks)]

    return blocks

