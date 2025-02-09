from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from keras.layers import Input, Conv2D, Conv3D, Activation, Lambda
from keras.models import Model
from keras.layers.merge import Add, Concatenate
import tensorflow as tf
from keras import backend as K
from .blocks import unet_block, unet_blocks, gaussian_2d
import re

from ..utils import _raise, backend_channels_last
import numpy as np


def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                n_channel_out=1,
                residual=False,
                prob_out=False,
                long_skip=True,
                eps_scale=1e-3):
    """ TODO """

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    # TODO: rewrite with conv_block
    conv = Conv2D if n_dim == 2 else Conv3D

    input = Input(input_shape, name="input")
    unet = unet_block(n_depth, n_filter_base, kernel_size, input_planes=input_shape[-1],
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size, long_skip=long_skip)(input)

    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    if residual:
        if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
            raise ValueError("number of input and output channels must be the same for a residual net.")
        final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final, scale])

    return Model(inputs=input, outputs=final)


def uxnet(input_shape,
          n_depth=2,
          n_filter_base=16,
          kernel_size=(3, 3),
          n_conv_per_depth=2,
          activation="relu",
          last_activation='linear',
          batch_norm=False,
          dropout=0.0,
          pool_size=(2, 2),
          residual=True,
          odd_to_even=False,
          shortcut=None,
          shared_idx=[],
          prob_out=False,
          eps_scale=1e-3):

    """
    Multi-body U-Net which learns identity by leaving one plane out in each branch

    :param input_shape:
    :param n_depth:
    :param n_filter_base:
    :param kernel_size:
    :param n_conv_per_depth:
    :param activation:
    :param last_activation:
    :param batch_norm:
    :param dropout:
    :param pool_size:
    :param prob_out:
    :param eps_scale:
    :return: Model
    """
    # TODO: fill params
    # TODO: add odd-to-even mode

    # Define vars
    channel_axis = -1 if backend_channels_last() else 1
    n_planes = input_shape[channel_axis]
    if n_planes % 2 != 0 and odd_to_even:
        raise ValueError('Odd-to-even mode does not support uneven number of planes')
    n_dim = len(kernel_size)
    conv = Conv2D if n_dim == 2 else Conv3D

    # Define functional model
    input = Input(shape=input_shape, name='input_main')

    # TODO test new implementation and remove old
    # Split planes (preserve channel)
    input_x = [Lambda(lambda x: x[..., i:i+1], output_shape=(None, None, 1))(input) for i in range(n_planes)]

    # We can train either in odd-to-even mode or in LOO mode
    if odd_to_even:
        # In this mode we stack together odd and even planes, train the net to predict even from odd and vice versa
        # input_x_out = [Concatenate(axis=-1)(input_x[j::2]) for j in range(2)]
        input_x_out = [Concatenate(axis=-1)(input_x[j::2]) for j in range(1, -1, -1)]
    else:
        # Concatenate planes back in leave-one-out way
        input_x_out = [Concatenate(axis=-1)([plane for i, plane in enumerate(input_x) if i != j]) for j in range(n_planes)]

    # if odd_to_even:
    #     input_x_out = [Lambda(lambda x: x[..., j::2],
    #                           output_shape=(None, None, n_planes // 2),
    #                           name='{}_planes'.format('even' if j == 0 else 'odd'))(input)
    #                    for j in range(1, -1, -1)]
    # else:
    #     # input_x_out = [Lambda(lambda x: x[..., tf.convert_to_tensor([i for i in range(n_planes) if i != j], dtype=tf.int32)],
    #     #                       output_shape=(None, None, n_planes-1),
    #     #                       name='leave_{}_plane_out'.format(j))(input)
    #     #                for j in range(n_planes)]
    #
    #     input_x_out = [Lambda(lambda x: K.concatenate([x[..., :j], x[..., (j+1):]], axis=-1),
    #                           output_shape=(None, None, n_planes - 1),
    #                           name='leave_{}_plane_out'.format(j))(input)
    #         for j in range(n_planes)]

    # U-Net parameters depend on mode (odd-to-even or LOO)
    n_blocks = 2 if odd_to_even else n_planes
    input_planes = n_planes // 2 if odd_to_even else n_planes-1
    output_planes = n_planes // 2 if odd_to_even else 1

    # Create U-Net blocks (by number of planes)
    unet_x = unet_blocks(n_blocks=n_blocks, input_planes=input_planes, output_planes=output_planes,
                         n_depth=n_depth, n_filter_base=n_filter_base, kernel_size=kernel_size,
                         activation=activation, dropout=dropout, batch_norm=batch_norm,
                         n_conv_per_depth=n_conv_per_depth, pool=pool_size, shared_idx=shared_idx)
    unet_x = [unet(inp_out) for unet, inp_out in zip(unet_x, input_x_out)]

    # Version without weight sharing:
    # unet_x = [unet_block(n_depth, n_filter_base, kernel_size,
    #                      activation=activation, dropout=dropout, batch_norm=batch_norm,
    #                      n_conv_per_depth=n_conv_per_depth, pool=pool_size,
    #                      prefix='out_{}_'.format(i))(inp_out) for i, inp_out in enumerate(input_x_out)]

    # TODO: rewritten for sharing -- remove commented below
    # Convolve n_filter_base to 1 as each U-Net predicts a single plane
    # unet_x = [conv(1, (1,) * n_dim, activation=activation)(unet) for unet in unet_x]

    if residual:
        if odd_to_even:
            # For residual U-Net sum up output for odd planes with even planes and vice versa
            unet_x = [Add()([unet, inp]) for unet, inp in zip(unet_x, input_x[::-1])]
        else:
            # For residual U-Net sum up output with its neighbor (next for the first plane, previous for the rest
            unet_x = [Add()([unet, inp]) for unet, inp in zip(unet_x, [input_x[1]]+input_x[:-1])]

    # Concatenate outputs of blocks, should receive (None, None, None, n_planes)
    # TODO assert to check shape?

    if odd_to_even:
        # Split even and odd, assemble them together in the correct order
        # TODO tests
        unet_even = [Lambda(lambda x: x[..., i:i+1],
                            output_shape=(None, None, 1),
                            name='even_{}'.format(i))(unet_x[0]) for i in range(n_planes // 2)]
        unet_odd = [Lambda(lambda x: x[..., i:i+1],
                           output_shape=(None, None, 1),
                           name='odd_{}'.format(i))(unet_x[1]) for i in range(n_planes // 2)]

        unet_x = list(np.array(list(zip(unet_even, unet_odd))).flatten())

    unet = Concatenate(axis=-1)(unet_x)

    if shortcut is not None:
        # We can create a shortcut without long skip connection to prevent noise memorization
        if shortcut == 'unet':
            shortcut_block = unet_block(long_skip=False, input_planes=n_planes,
                                       n_depth=n_depth, n_filter_base=n_filter_base, kernel_size=kernel_size,
                                       activation=activation, dropout=dropout, batch_norm=batch_norm,
                                       n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)
            shortcut_block = conv(n_planes, (1,) * n_dim, activation='linear', name='shortcut_final_conv')(shortcut_block)

        # Or a simple gaussian blur block
        elif shortcut == 'gaussian':
            shortcut_block = gaussian_2d(n_planes, k=13, s=7)(input)

        else:
            raise ValueError('Shortcut should be either unet or gaussian')

        # TODO add or concatenate?
        unet = Add()([unet, shortcut_block])
        # unet = Concatenate(axis=-1)([unet, shortcut_unet])



    # Final activation layer
    final = Activation(activation=last_activation)(unet)

    if prob_out:
        scale = conv(n_planes, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final, scale])

    return Model(inputs=input, outputs=final)


def common_unet(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1,
                residual=True, prob_out=False, long_skip=True, last_activation='linear'):
    """
    Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_
    to be used for image restoration/enhancement.

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out : bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)
    last_activation : str
        name of activation function for the final output layer

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim, pool_size=(2,)*n_dim,
                           n_channel_out=n_channel_out, residual=residual, prob_out=prob_out, long_skip=long_skip)
    return _build_this


def common_uxnet(n_dim=2, n_depth=1, kern_size=3, n_first=16,
                 residual=True, prob_out=False, last_activation='linear',
                 shared_idx=[], odd_to_even=False, shortcut=None):
    def _build_this(input_shape):
        return uxnet(input_shape=input_shape, last_activation=last_activation, n_depth=n_depth, n_filter_base=n_first,
                     kernel_size=(kern_size,)*n_dim, pool_size=(2,)*n_dim,
                     residual=residual, prob_out=prob_out,
                     shared_idx=shared_idx, odd_to_even=odd_to_even, shortcut=shortcut)
    return _build_this


modelname = re.compile("^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$")


def common_unet_by_name(model):
    r"""Shorthand notation for equivalent use of :func:`common_unet`.

    Parameters
    ----------
    model : str
        define model to be created via string, which is parsed as a regular expression:
        `^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$`

    Returns
    -------
    function
        Calls :func:`common_unet` with the respective parameters.

    Raises
    ------
    ValueError
        If argument `model` is not a valid string according to the regular expression.

    Example
    -------
    >>> model = common_unet_by_name('resunet2_1_3_16_1out')(input_shape)
    >>> # equivalent to: model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    Todo
    ----
    Backslashes in docstring for regexp not rendered correctly.

    """
    m = modelname.fullmatch(model)
    if m is None:
        raise ValueError("model name '%s' unknown, must follow pattern '%s'" % (model, modelname.pattern))
    # from pprint import pprint
    # pprint(m.groupdict())
    options = {k:int(m.group(k)) for k in ['n_depth','n_first','kern_size']}
    options['prob_out'] = m.group('prob_out') is not None
    options['residual'] = {'unet': False, 'resunet': True}[m.group('model')]
    options['n_dim'] = int(m.group('n_dim'))
    options['n_channel_out'] = 1 if m.group('n_channel_out') is None else int(m.group('n_channel_out'))
    if m.group('last_activation') is not None:
        options['last_activation'] = m.group('last_activation')

    return common_unet(**options)


def receptive_field_unet(n_depth, kern_size, pool_size=2, n_dim=2, img_size=1024):
    """Receptive field for U-Net model (pre/post for each dimension)."""
    x = np.zeros((1,)+(img_size,)*n_dim+(1,))
    mid = tuple([s//2 for s in x.shape[1:-1]])
    x[(slice(None),) + mid + (slice(None),)] = 1
    model = custom_unet (
        x.shape[1:],
        n_depth=n_depth, kernel_size=[kern_size]*n_dim, pool_size=[pool_size]*n_dim,
        n_filter_base=8, activation='linear', last_activation='linear',
    )
    y  = model.predict(x)[0,...,0]
    y0 = model.predict(0*x)[0,...,0]
    ind = np.where(np.abs(y-y0)>0)
    return [(m-np.min(i), np.max(i)-m) for (m, i) in zip(mid, ind)]