from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, backend_channels_last

import numpy as np
import keras.backend as K


def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x, axis=-1)) if mean else (lambda x: x)


def loss_laplace(mean=True):
    R = _mean_or_not(mean)
    C = np.log(2.0)
    if backend_channels_last():
        def nll(y_true, y_pred):
            n = K.shape(y_true)[-1]
            mu = y_pred[..., :n]
            sigma = y_pred[..., n:]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll
    else:
        def nll(y_true, y_pred):
            n = K.shape(y_true)[1]
            mu = y_pred[:, :n, ...]
            sigma = y_pred[:, n:, ...]
            return R(K.abs((mu-y_true)/sigma) + K.log(sigma) + C)
        return nll


def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[..., :n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:, :n, ...] - y_true))
        return mae


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[..., :n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:, :n, ...] - y_true))
        return mse


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss


def loss_ssim(k1=0.01, k2=0.03, ksize=3, stride=2, rate=1, padding='VALID'):
    # TODO: implement channels first as well, currently only channels last

    def ssim(y_true, y_pred):
        # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b
        # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
        c1 = k1 ** 2
        c2 = k2 ** 2

        # TODO: remove hardcoded values -- currently patch extractor cannot work with None shapes
        # y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))
        # y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_true)[1:]))

        y_pred = K.reshape(y_pred, [-1, 64, 64, 9])
        y_true = K.reshape(y_true, [-1, 64, 64, 9])

        patches_true = K.tf.extract_image_patches(
            images=y_true,
            ksizes=[1, ksize, ksize, 1],  # size of the sliding window for each dimension
            strides=[1, stride, stride, 1],  # how far the centers of two consecutive patches
            rates=[1, rate, rate, 1],  # how far two consecutive patch samples
            padding=padding)

        patches_pred = K.tf.extract_image_patches(
            images=y_pred,
            ksizes=[1, ksize, ksize, 1],  # size of the sliding window for each dimension
            strides=[1, stride, stride, 1],  # how far the centers of two consecutive patches
            rates=[1, rate, rate, 1],  # how far two consecutive patch samples
            padding=padding)

        mu_true = K.mean(patches_true, axis=-1)
        mu_pred = K.mean(patches_pred, axis=-1)

        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)

        # std_true = K.sqrt(var_true)
        # std_pred = K.sqrt(var_pred)

        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - mu_true * mu_pred

        # ssim = (2 * mu_true * mu_pred + c1) * (2 * std_pred * std_true + c2)
        ssim = (2 * mu_true * mu_pred + c1) * (2 * covar_true_pred + c2)
        ssim /= (mu_true ** 2 + mu_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim = K.tf.where(K.tf.is_nan(ssim), K.zeros_like(ssim), ssim)  # replace NaN with zeros

        return K.mean(((1.0 - ssim) / 2))

    return ssim
