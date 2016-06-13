#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pdb import set_trace
import tensorflow as tf
from operator import mul
import re
strides = [
    None,
    [1, 1, 1, 1],
    [1, 2, 2, 1],
    ]
relu = tf.nn.relu


def fname(layer, name):
    if layer:
        return '%s/%s' % (layer, name)
    else:
        return name


def conv_layer(inpt, fsize, chnl, nstride, padding='SAME',
               layer='', reuse=None):
    with tf.variable_scope(layer, reuse=reuse):
        stride = strides[nstride]
        c0 = inpt.get_shape().as_list()[-1]
        fsize = [3, 3, c0, chnl]
        # filter1 = tf.Variable(
        #     tf.random_normal(fsize),
        #     name=fname(layer, 'conv'),
        #     trainable=True)
        filter1 = tf.get_variable(
            name='conv',
            shape=fsize,
            dtype=np.float32,
            initializer=tf.random_normal_initializer(),
            trainable=True)
        ret = tf.nn.conv2d(inpt, filter1, strides=stride, padding=padding)
    return ret


def maxpool_layer(inpt, msize, nstride, padding='SAME'):
    # conv = maxpool_layer(conv, [3, 3], 2)
    msize = [1, msize[0], msize[1], 1]
    stride = strides[nstride]
    return tf.nn.max_pool(inpt, msize, stride, padding)


def avgpool_layer(inpt, msize, nstride, padding='SAME'):
    # conv = avgpool_layer(conv, [3, 3], 2)
    msize = [1, msize[0], msize[1], 1]
    stride = strides[nstride]
    return tf.nn.avg_pool(inpt, msize, stride, padding, name='avgpool')


def red_mul(xs):
    return reduce(mul, xs)


def full_layer(inpt, ndim, layer='', reuse=None):
    with tf.variable_scope(layer, reuse=reuse):
        size0 = getshape(inpt)
        size1 = [size0[0], red_mul(size0[1:])]
        newinpt = tf.reshape(inpt, size1)
        # weight = tf.Variable(
        #     tf.random_normal([size1[1], ndim]),
        #     name=fname(layer, 'weight'),
        #     trainable=True)
        # bias = tf.Variable(
        #     tf.zeros([ndim]),
        #     name=fname(layer, 'bias'),
        #     trainable=True)
        weight = tf.get_variable(
            name='weight',
            shape=[size1[1], ndim],
            dtype=np.float32,
            initializer=tf.random_normal_initializer(),
            trainable=True)
        bias = tf.get_variable(
            name='bias',
            dtype=np.float32,
            initializer=tf.zeros_initializer(shape=[ndim]),
            trainable=True)
        # tf.get_variable_scope().reuse_variables()
        ret = tf.matmul(newinpt, weight) + bias
    return ret


def getshape(x):
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        return x.get_shape().as_list()
    elif x.type == 'Placeholder':
        return [s.size for s in x.get_attr('shape').dim]
    else:
        raise Exception('Unknown type')


def listVar(tensor, namescope=None, coll_key='trainable_variables'):
    if namescope is None:
        namescope = getscope(tensor)
    key = '^%s' % namescope
    li = []
    for v in tensor.graph._collections[coll_key]:
        rets = re.search(key, v.name)
        if rets:
            li.append(v)
    return li


def listPlaceholder(tensor):
    for v in tensor.graph.get_operations():
        if v.type == 'Placeholder':
            yield v


def getscope(tensor):
    try:
        return re.findall('.*/', tensor.name)[0]
    except:
        return ''


def compare(model1, model2):
    fsize = lambda v: tuple(v.get_shape().as_list())
    scope1 = getscope(model1)
    scope2 = getscope(model2)
    iter1 = listVar(model1)
    iter2 = listVar(model2)
    for v1, v2 in zip(iter1, iter2):
        assert v1.name.replace(scope1, '') == v2.name.replace(scope2, '')
        assert fsize(v1) == fsize(v2)


def setAssign(model1, model2):
    fiter = lambda model: listVar(model)
    li = []
    for v1, v2 in zip(*map(fiter, (model1, model2))):
        r = v2.assign(v1)
        li.append(r)
    return li


def getidx(mat, acts):
    # Get element value by index with each row
    nrow, ncol = mat.get_shape().as_list()
    mat_1d = tf.reshape(mat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, acts)
    ret = tf.gather(mat_1d, idx)
    return ret


def test():
    tf.reset_default_graph()

    def subinit(state, scope='', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            model = relu(conv_layer(
                state, [2, 2], 16, 1, layer='clayer1', reuse=reuse))
            model = relu(conv_layer(
                model, [2, 2], 32, 1, layer='clayer2', reuse=reuse))
            model = full_layer(model, 16, layer='layer1', reuse=reuse)
            model = full_layer(model, 4, layer='layer2', reuse=reuse)
            # tf.get_variable_scope().reuse_variables=True
            return model

    size = [5, 4, 4, 3]
    state = tf.placeholder(tf.float32, shape=size)
    state2 = tf.placeholder(tf.float32, shape=size)

    model = subinit(state, 'a', reuse=None)
    model2 = subinit(state2, 'a', reuse=True)

    w1 = listVar(model, namescope='a/')[0]
    w2 = listVar(model2, namescope='a/')[0]

    initvar = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(initvar)

    w1v, w2v = sess.run([w1, w2])
    print w1v.sum() == w2v.sum(), (w1v.sum(), w2v.sum())

    x1 = np.random.randn(*size)
    x2 = np.random.randn(*size)
    r1, r2 = sess.run(
        [model, model2],
        feed_dict={
            state: x1,
            state2: x2,
            })
    print (r1.sum() != r2.sum()), (r1.sum(), r2.sum())

    r1, r2 = sess.run(
        [model, model2],
        feed_dict={
            state: x1,
            state2: x1,
            })
    print (r1.sum() == r2.sum()), (r1.sum(), r2.sum())

    for v in listPlaceHolder(model):
        print v.name, getshape(v)
    globals().update(locals())


if __name__ == '__main__':
    if 'sess' in globals():
        sess.close()
    test()
