#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle, gzip
import copy
from tftools import *
import os
import yaml
from traceback import print_exc
import itertools as it
import tensorflow as tf
import numpy as np
from pdb import set_trace
from random import random, randint
from utils import chkEmpty, StateAct, encState, encReward, zeros
import savedata
from operator import mul
from StateAct import SARli
SEED = 34654
N_BATCH = 100
N_REPSIZE = 500


def rndAction(state):
    try:
        return randint(0, 3)
    except:
        print_exc()
        set_trace()


class Model(object):
    def evalR(self, wl):
        """ evalaute reward given state """
        if wl == 0:
            return 0.
        elif wl == -1:
            return 1
        elif wl == self.sgn:
            return 2
        else:
            return -2

    def setScore(self, score):
        """"""

    def replay(self):
        """"""


class Random(Model):
    # Random Player
    def predict(self, state):
        return rndAction(state)

    def update(self, *args):
        return self

    def reset(self):
        """"""

    def save(self):
        """"""


class NNQ(Model):
    def __init__(self, **kwargs):
        algo = kwargs.get('algo', 'ANN')
        print('Use %s' % algo)
        self.SARs = SARli(lim=N_REPSIZE)  # List of (state, action)
        self.alpha = kwargs.get('alpha', 0.5)
        self.gamma = kwargs.get('gamma', 0.5)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.nolearn = not kwargs.get('train', True)
        self.kw = kwargs.get('kw', '')
        self.savegame = kwargs.get('savegame', False)
        self.trainNFQ = kwargs.get('trainNFQ', False)
        self.algo = algo
        self.score = 0
        self.sa0 = None

        with tf.variable_scope('original'):
            self.state = tf.placeholder(
                tf.float32, shape=(N_BATCH, 4, 4, 4), name='state')
            self.act = tf.placeholder(
                tf.int32, shape=[N_BATCH], name='act')
            self.r = tf.placeholder(
                tf.float32, shape=[N_BATCH, 1], name='r')
            self.state1 = tf.placeholder(
                tf.float32, shape=(N_BATCH, 4, 4, 4), name='state1')

        self.model_init = eval(algo)
        self.model = self.model_init(
            self.state, 'model', reuse=None)
        self.Qhat = self.model_init(
            self.state1, 'model', reuse=True)
        self.loss_def(
            self.act,
            self.r,
            self.gamma,
            )
        self.parms = listVar(self.model)

        for x in self.parms:
            tf.histogram_summary(x.name, x)

        self.ExpReplay_opt = ExpReplay(
            self, shuffle=kwargs['shuffle'])
        self.TargetNetwork_opt = TargetNetwork(
            self, enable=kwargs['TargetNetwork'])
        self.saver = tf.train.Saver(self.parms)

        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()

        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(init)
        self.writer_sum = tf.train.SummaryWriter(
            'tmp/%s' % kwargs['logdir'],
            self.sess.graph)
        self.tickcnt = 0

    def show(self):
        print 'Maybe deprecated'
        for i in range(len(self.SARs)):
            print i, self.SARs[i].score

    def listparm(self):
        cnt = 0
        for x in self.parms:
            size = getshape(x)
            print x.name, size
            cnt += reduce(mul, size)
        print 'Total parameters:', cnt

    def setgame(self, game):
        self.records = game.records

    def update(self, t, state, act, r, terminal):
        # receive state class
        if self.nolearn:
            return

        state = encState(state)
        self.SARs.update(t-1, state, act, r, terminal)
        return self

    def _update(self, Target=False, nlim=None):
        if self.nolearn:
            return
        if (len(self.SARs) < N_REPSIZE) or (self.nolearn):
            return None

        for ii in xrange(100):
            iter1 = self.ExpReplay_opt.getli(self.SARs)
            if nlim:
                iter1 = it.islice(iter1, nlim)

            li = []
            for t, (S, A, R, S1, terminals) in enumerate(iter1):
                parms = self.sess, S, A, R, S1, self.summary
                if Target:
                    ret = self.TargetNetwork_opt.optimize(*parms)
                else:
                    ret = self.optimize(*parms)

                if ret is not None:
                    loss_res, merge_res = ret
                    if loss_res:
                        li.append(loss_res)

                    if t and (t % 10 == 0):
                        self.tickcnt += 1
                        print t, self.MSE(self.sa0)
                        if merge_res is not None:
                            self.writer_sum.add_summary(
                                merge_res, self.tickcnt)
                        if (self.tickcnt % 10 == 0) and (not Target):
                            self._update(Target=True)
        if len(li) == 0:
            return None
        else:
            return np.mean(li)

    def predict(self, state):
        # state = encState(state)
        """ epsilon-greedy algorithm """
        if (len(self.records) < 3):
            return rndAction(state)

        state = self.SARs.getstate_new(self.encState(state))
        if random() < self.epsilon:
            act = rndAction(state)
        else:
            act = self._action(state)
        return act

    def _action(self, state):
        """
        Return best action given state and win/loss
        Like predict process in NN
        Best action: argmax( R(s, a) + gamma * max(R(s', a')) )
        """
        act = -1
        assert type(state) == np.ndarray, type(state)
        state = self.FULL(state)
        r, = self.sess.run(
            [self.model],
            feed_dict={self.state: state})

        for i in np.argsort(r[0, :].ravel())[::-1]:
            act = i
            break
        assert act != -1
        return act

    def getparm(self):
        return [(p.name, val) for p, val in
                zip(self.parms, self.sess.run(self.parms))]

    def reset(self):
        self.score = 0

    def save(self):
        if self.nolearn:
            return
        # self.saveobj.save(self.getparm())
        self.saver.save(
            self.sess,
            'tmp/%s%s.ckpt' % (self.algo, self.kw)
            )
        return self

    def load(self):
        fi = 'tmp/%s%s.ckpt' % (self.algo, self.kw)
        if os.path.exists(fi):
            print 'Load parameters'
            self.saver.restore(self.sess, fi)
        return self

    def FULL(self, s0):
        n = N_BATCH - s0.shape[0]
        return np.vstack([s0, zeros(n)])

    def encState(self, state, noadd=False):
        s1 = []
        if not noadd:
            [s1.extend(copy.deepcopy(r[0])) for r in self.records[-3:]]
        s1.extend(copy.deepcopy(state))
        assert len(s1) == 16
        return encState(s1)

    def loss_def(self, act, r, gamma):
        with tf.variable_scope('original'):
            Qsa = fgetidx(self.model, self.act)
            loss = r + gamma * maxQ(self.Qhat) - Qsa
            self.loss = tf.reduce_mean(tf.square(loss))
            self.op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def optimize(self, sess, state, act, r, state1, summary=None):
        varlist = [self.loss, self.op]
        if summary is not None:
            varlist.append(summary)
        feed_dict = {
            self.state: state,
            self.act: act,
            self.r: r,
            self.state1: state1,
            }

        try:
            ret = sess.run(
                varlist,
                feed_dict=feed_dict
                )
        except:
            print_exc()
            set_trace()

        if summary is not None:
            return ret[0], ret[2]
        else:
            return ret[0]

    def MSE(self, SARs):
        def fun(i):
            idx = range(i, i+N_BATCH)
            if idx[-1] >= len(SARs):
                return np.nan
            state, act, r, state1, terminal = SARs[idx]
            loss = self.sess.run(
                self.loss,
                feed_dict={
                    self.state: state,
                    self.act: act,
                    self.r: r,
                    self.state1: state1
                    }
                )
            return loss
        li = map(fun, xrange(0, len(SARs), N_BATCH))
        return np.nanmean(li)


def ANN(state, layer='', reuse=None):
    with tf.variable_scope(layer, reuse=reuse):
        # model = relu(full_layer(state, 256, layer='layer1', reuse=reuse))
        model = relu(full_layer(state, 64, layer='layer1', reuse=reuse))
        model = relu(full_layer(model, 16, layer='layer2', reuse=reuse))
        model = relu(full_layer(model, 4, layer='layer3', reuse=reuse))
    return model


def CNN(state, layer='', reuse=None):
    with tf.variable_scope(layer, reuse=reuse):
        model = relu(conv_layer(
            state, [2, 2], 16, 1, layer='layer1', reuse=reuse))
        model = relu(conv_layer(
            model, [2, 2], 32, 1, layer='layer2', reuse=reuse))
        model = relu(full_layer(model, 256, layer='layer3', reuse=reuse))
        model = relu(full_layer(model, 4, layer='layer4', reuse=reuse))
    return model


def getidx(mat, act):
    # Get element value by index with each row
    nrow, ncol = mat.get_shape().as_list()
    mat_1d = tf.reshape(mat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, act)
    ret = tf.gather(mat_1d, idx)
    return ret


fgetidx = tf.make_template('getidx', getidx)


def maxQ(Q):
    Q_max = tf.reduce_max(Q, reduction_indices=[1])
    size0 = getshape(Q_max)
    return tf.reshape(Q_max, [size0[0], 1])


class ExpReplay(object):
    def __init__(self, obj, shuffle=False):
        self.shuffle = shuffle

    def getli(self, SARs):
        N = len(SARs)
        if self.shuffle:
            idxs = np.random.permutation(range(N))
        else:
            idxs = range(N)
        for t in xrange(0, N, N_BATCH):
            idx = idxs[t:(t+N_BATCH)]
            if len(idx) == N_BATCH:
                yield SARs[idx]


class TargetNetwork(object):
    """
    - Use old weight to calculate:
        Q0(r, gamma, s1, Q_0) = r + gamma * max_a Q_0(s1, a)
    - Loss Tensor:
        Loss(Q0, s0, act, Q) = mean(sqrt(Q0 - Q(s0, act)))
    - Minimize Loss:
        Optim(Q0, s0, act, Q) = tf.train.AnyOptimizer(Loss)
    """
    def __init__(self, obj, enable=False):
        self.state = obj.state
        self.act = obj.act
        self.r = obj.r
        self.state1 = obj.state1
        self.enable = enable

        # initialize another NN structure
        self.model_old = obj.model_init(self.state1, 'model0')
        self.assigns = setAssign(obj.model, self.model_old)
        self.firstRun = False

        with tf.variable_scope('new'):
            nrow, ncol = getshape(self.model_old)
            self.Qval_old = tf.placeholder(
                self.model_old.dtype,
                shape=[nrow, 1],
                name='Qval',
                )
            self.Qhat_old = self.r + obj.gamma * maxQ(self.model_old)
            Qsa = fgetidx(obj.model, self.act)
            self.loss = tf.reduce_mean(
                tf.square(self.Qval_old - Qsa)
                )
            self.op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def optimize(self, sess, state, act, r, state1, summary=None):
        if not self.enable:
            return

        if not self.firstRun:
            sess.run(self.assigns)
            self.firstRun = True
            return

        Qhat = sess.run(
            self.Qhat_old,
            feed_dict={
                self.r: r,
                self.state1: state1,
                })

        varlist = [self.loss, self.op]

        ret = sess.run(
            varlist,
            feed_dict={
                self.Qval_old: Qhat,
                self.state: state,
                self.act: act,
                }
            )

        sess.run(self.assigns)

        return ret[0], None


def NFQ(**kwargs):
    saveflag = True
    agent = NNQ(**kwargs)
    # agent.load()
    agent.listparm()
    SARs = agent.SARs
    SARs.lim = 2000

    nlim = 20000
    train = range(188)
    test = range(188, 363)
    ALL = range(nlim)
    flag = train
    li = []
    sa0 = None

    for i, fi in enumerate(os.listdir('data')):
        if i not in flag:
            continue
        fi = 'data/%s' % fi
        rets = cPickle.load(gzip.open(fi))

        for j, (t, state, act, r, terminal) in enumerate(rets):
            if r >= 0:
                r += 2
            agent.update(t, state, act, r, terminal)
            if len(SARs) >= 1001:
                break

        if len(SARs) >= 1001:
            if (sa0 is None) and (len(SARs) > 1000):
                sa0 = SARs.subset(0, 1000)
                agent.sa0 = sa0
            loss = agent._update()

            if loss is not None:
                loss_mse = agent.MSE(sa0)
                print loss_mse
                li.append(loss_mse)
                np.savez(open('%s.npz' % kwargs['kw'], 'wb'), li)
            break
        agent.reset()

    # loss = agent.replay()
    # print i, np.mean(loss), len(SARs)
    if saveflag:
        agent.save()


def test():
    elems = tf.Variable(np.arange(10, dtype=np.float32))
    elems = tf.identity(elems)
    op_sum = tf.foldl(lambda a, x: a + x, elems)
    op_mean = tf.reduce_mean(elems)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print sess.run([op_sum, op_mean])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file')
    args = parser.parse_args()
    args = vars(args)
    if args['config']:
        config = yaml.load(open(args['config'], 'rb'))
        args.update(config)
    if args.get('agent'):
        args['agent'] = args['agent'].lower()
    print(args)

    if args.get('N_BATCH'):
        N_BATCH = args['N_BATCH']
    if args.get('N_REPSIZE'):
        N_REPSIZE = args['N_REPSIZE']

    NFQ(**args)
