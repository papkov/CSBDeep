from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last

from keras.layers import Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D
from keras.layers.merge import Concatenate
from keras.models import Sequential


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


def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3, 3), n_conv_per_depth=2,
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

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    conv = Conv2D if n_dim == 2 else Conv3D
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        if shared_middle is None:
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   name=_name("middle_%s" % i))(layer)
            # TODO should it be n_conv_per_depth-1 for name?
            layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation,
                               batch_norm=batch_norm,
                               name=_name("middle_%s" % n_conv_per_depth))(layer)
        else:
            layer = shared_middle(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   batch_norm=batch_norm,
                                   name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm,
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

    # TODO Implement depth, batch norm, padding, init and dropout selection for the middle layer
    # Now depth 2 is hardcoded, no batch norm and dropout
    def get_shared_middle():
        n_dim = len(kernel_size)
        # conv_block = conv_block2 if n_dim == 2 else conv_block3
        #
        # def _name(s):
        #     return prefix + s

        layer = Sequential([Conv2D(n_filter_base * 2 ** n_depth, kernel_size,
                                   activation=activation,
                                   padding='same',
                                   kernel_initializer='glorot_uniform',
                                   input_shape=(None,)*n_dim + (n_filter_base * 2 ** max(0, n_depth - 1),),
                                   name='middle_0'),
                            Conv2D(n_filter_base * 2 ** max(0, n_depth - 1), kernel_size,
                                   activation=activation,
                                   padding='same',
                                   kernel_initializer='glorot_uniform',
                                   name='middle_1')])

        # for i in range(n_conv_per_depth - 1):
        #     layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
        #                        dropout=dropout,
        #                        activation=activation,
        #                        batch_norm=batch_norm,
        #                        name=_name("middle_%s" % i))(layer)
        #
        #
        # layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
        #                    dropout=dropout,
        #                    activation=activation,
        #                    batch_norm=batch_norm,
        #                    name=_name("middle_%s" % n_conv_per_depth))(layer)
        return layer

    # Build shared middle layer
    shared_middle = get_shared_middle() if share_middle else None
    if share_middle:
        print('Share middle layer')
    else:
        print('Do not share middle layer')

    blocks = [unet_block(n_depth=n_depth,
                         n_filter_base=n_filter_base,
                         kernel_size=kernel_size,
                         n_conv_per_depth=n_conv_per_depth,
                         activation=activation,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         last_activation=last_activation,
                         shared_middle=shared_middle,
                         pool=pool,
                         prefix='{}U{}_'.format(prefix, i))
              for i in range(n_blocks)]

    return blocks

