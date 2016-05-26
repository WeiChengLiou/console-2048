#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from operator import mul
relu = tf.nn.relu
softmax = tf.nn.softmax
strides = [
    None,
    [1, 1, 1, 1],
    [1, 2, 2, 1],
    ]


def add_conv(inpt, fsize, chnl, nstride, padding='SAME'):
    stride = strides[nstride]
    c0 = inpt.get_shape().as_list()[-1]
    fsize = [3, 3, c0, chnl]
    filter1 = tf.Variable(
        tf.random_normal(fsize),
        name='conv_filter',
        trainable=True,
        )
    return tf.nn.conv2d(inpt, filter1, strides=stride, padding=padding)


def add_maxpool(inpt, msize, nstride, padding='SAME'):
    # conv = add_maxpool(conv, [3, 3], 2)
    msize = [1, msize[0], msize[1], 1]
    stride = strides[nstride]
    return tf.nn.max_pool(inpt, msize, stride, padding)


def add_avgpool(inpt, msize, nstride, padding='SAME'):
    # conv = add_avgpool(conv, [3, 3], 2)
    msize = [1, msize[0], msize[1], 1]
    stride = strides[nstride]
    return tf.nn.avg_pool(inpt, msize, stride, padding)


def red_mul(xs):
    return reduce(mul, xs)


def add_fullcon(inpt, ndim, normfun=None):
    # normfun example:
    #     lambda size: tf.truncated_normal(size, stddev=0.1, seed=SEED)
    size0 = getshape(inpt)
    size1 = [size0[0], red_mul(size0[1:])]
    newinpt = tf.reshape(inpt, size1)

    if normfun is None:
        normfun = lambda size: tf.random_normal(size)

    weight = tf.Variable(
        normfun([size1[1], ndim]),
        name='weight',
        trainable=True)
    bias = tf.Variable(
        tf.zeros([ndim]),
        name='bias',
        trainable=True)
    return tf.matmul(newinpt, weight) + bias


def getshape(x):
    return x.get_shape().as_list()
