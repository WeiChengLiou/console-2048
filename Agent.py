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
from random import random, randint, sample
from utils import encState, zeros, yload, ysave
from operator import mul
from StateAct import SARli
SEED = 34654
N_BATCH = 100
N_REPSIZE = 500
OPTIMIZER = tf.train.AdamOptimizer
LEARNING_RATE = 1e-3
TARGET_FREQ = 100


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


class DNQ(Model):
    def __init__(self, **kwargs):
        algo = kwargs.get('algo', 'ANN')
        print('Use %s' % algo)
        self.SARs = SARli(lim=N_REPSIZE)  # List of (state, action)
        self.gamma = kwargs.get('gamma', 0.5)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.nolearn = not kwargs.get('train', True)
        self.kw = kwargs.get('kw', '')
        self.savegame = kwargs.get('savegame', False)
        self.trainNFQ = kwargs.get('trainNFQ', False)
        self.algo = algo
        self.score = 0
        self.sa_sample = None

        with tf.variable_scope('original'):
            self.state = tf.placeholder(
                tf.float32, shape=(N_BATCH, 4, 4, 1), name='state')
            self.act = tf.placeholder(
                tf.int32, shape=[N_BATCH], name='act')
            self.target = tf.placeholder(
                tf.float32, shape=[N_BATCH, 1], name='r')

        self.model_init = eval(algo)
        self.model = self.model_init(
            self.state, 'model', reuse=None)
        self.loss_def(
            self.model,
            self.act,
            self.target,
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
        return self._update()

    def _update(self, Target=False):
        if self.nolearn:
            return
        if (len(self.SARs) < N_REPSIZE):
            return

        loss = None
        iter1 = self.ExpReplay_opt.getli(self.SARs, n_run=2)
        runTarget = False
        for t, (S, A, R, S1, terminals) in enumerate(iter1):
            if Target:
                self.TargetNetwork_opt.optimize(
                    self.sess, S, A, R, S1, terminals)
                continue
            else:
                self.tickcnt += 1
                ret = self.optimize(
                    S, A, R, S1, terminals,
                    summary=True)
            assert ret is not None

            if self.tickcnt % 10 == 0:
                loss = self.fitness(self.sa_sample)

            if (not Target) and (self.tickcnt % TARGET_FREQ == 0):
                if (self.TargetNetwork_opt.enable):
                    runTarget = True

        if runTarget:
            self._update(True)

        if (not Target) and (loss is None):
            loss = self.fitness(self.sa_sample)

        return loss

    def predict(self, state):
        # state = encState(state)
        """ epsilon-greedy algorithm """
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
        Q, = self.eval(state)
        Q = Q[0, :]

        for i in np.argsort(Q.ravel())[::-1]:
            act = i
            break
        assert act != -1
        return act

    def getparm(self):
        return [(p.name, val) for p, val in
                zip(self.parms, self.sess.run(self.parms))]

    def reset(self):
        self.score = 0

    def save(self, idx=None):
        if self.nolearn:
            return
        # self.saveobj.save(self.getparm())
        if idx:
            fi = 'tmp/%s%s.%s.ckpt' % (self.algo, self.kw, idx)
        else:
            fi = 'tmp/%s%s.ckpt' % (self.algo, self.kw)
        print fi

        self.saver.save(self.sess, fi)
        return self

    def load(self, idx=None):
        if idx:
            fi = 'tmp/%s%s.%s.ckpt' % (self.algo, self.kw, idx)
        else:
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

    def loss_def(self, model, act, target, gamma):
        with tf.variable_scope('original'):
            Qsa = tf.reshape(fgetidx(model, act), [-1, 1])
            loss = target - Qsa
            loss = tf.square(loss)
            self.loss = tf.reduce_mean(loss)
            self.op = OPTIMIZER(LEARNING_RATE).minimize(self.loss)
            self.loss_i = loss
            self.Qsa = Qsa

    def eval(self, state, getmax=False):
        Q = self.sess.run(
            self.model,
            feed_dict={
                self.state: state
                })
        if getmax:
            return np.max(Q, axis=1).reshape([-1, 1])
        else:
            return Q

    def optimize(self, state, act, r, state1, terminals,
                 summary=False):
        target = r
        if self.gamma:
            Qhat = self.eval(state1, getmax=True)
            idx = np.argwhere(terminals == 0).ravel()
            target[idx, 0] += self.gamma * Qhat[idx, 0]

        return self._optfun(
            state, act, target,
            summary=summary)

    def _optfun(self, state, act, target,
                summary=False):
        varlist = [self.loss, self.op]
        if summary:
            varlist.append(self.summary)

        feed_dict = {
            self.state: state,
            self.act: act,
            self.target: target,
            }

        ret = self.sess.run(
            varlist,
            feed_dict=feed_dict
            )

        if summary and (self.tickcnt % 10 == 0):
            merge_res = ret[-1]
            self.writer_sum.add_summary(
                merge_res, self.tickcnt/10)

        return ret[0]

    def fitness(self, ret):
        N = len(ret[0])

        def fun(i):
            idx = range(i, i+N_BATCH)
            if idx[-1] >= N:
                return np.nan
            state = ret[0][idx]
            loss = self.eval(state, getmax=True)
            return loss
        li = map(fun, xrange(0, N, N_BATCH))
        return np.nanmean(li)


def ANN(state, layer='', reuse=None):
    std = 0.01
    with tf.variable_scope(layer, reuse=reuse):
        model = relu(full_layer(
            state, 256, layer='layer1', reuse=reuse, stddev=std))
        model = relu(full_layer(
            model, 256, layer='layer2', reuse=reuse, stddev=std))
        model = relu(full_layer(
            model, 4, layer='layer3', reuse=reuse, stddev=std))
    return model


def CNN(state, layer='', reuse=None):
    std = 0.01
    with tf.variable_scope(layer, reuse=reuse):
        model = (conv_layer(
            state, [2, 2], 16, 1, layer='layer1', reuse=reuse, stddev=std))
        model = (conv_layer(
            model, [2, 2], 32, 1, layer='layer2', reuse=reuse, stddev=std))
        model = relu(full_layer(
            model, 64, layer='layer3', reuse=reuse, stddev=std))
        model = tf.tanh(full_layer(
            model, 4, layer='layer4', reuse=reuse, stddev=std))
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

    def getli(self, SARs, n_run=10):
        N = len(SARs)
        idxs = range(N)
        if self.shuffle:
            idxs = sample(idxs, N_BATCH*n_run)

        for t in xrange(0, len(idxs), N_BATCH):
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
        self.target = obj.target
        self.enable = enable
        self.gamma = obj.gamma

        # initialize another NN structure
        self._optfun = obj._optfun
        self.model_old = obj.model_init(self.state, 'model0')
        self.assigns = setAssign(obj.model, self.model_old)
        self.firstRun = False

    def optimize(self, sess, state, act, r, state1, terminals):
        if not self.enable:
            return

        if not self.firstRun:
            sess.run(self.assigns)
            self.firstRun = True
            return

        target = r
        if self.gamma:
            Qhat = sess.run(
                self.model_old,
                feed_dict={self.state: state1})
            Qhat = np.max(Qhat, axis=1).reshape([-1, 1])
            idx = np.argwhere(terminals == 0).ravel()
            target[idx, 0] += self.gamma * Qhat[idx, 0]
        self._optfun(state, act, target, summary=False)

        sess.run(self.assigns)


def NFQ(**kwargs):
    saveflag = False
    agent = DNQ(**kwargs)
    # agent.load()
    agent.listparm()

    if os.path.exists('perfdic.yaml'):
        perfdic = yload()
        idx = np.max(perfdic.keys()) + 1
    else:
        perfdic = {}
        idx = 0

    def callupdate(tick):
        t, state, act, r1, terminal = tick
        loss = agent.update(t, state, act, r1, terminal)
        return loss

    def gettick():
        cnt = 0
        for fi in sorted(os.listdir('data')):
            fi = 'data/%s' % fi
            rets = cPickle.load(gzip.open(fi))

            for (t, state, act, r, terminal) in rets:
                if terminal:
                    r = -1
                elif r > 0:
                    r = 1
                cnt += 1
                yield cnt, (t, state, act, r, terminal)

    try:
        n = 2000
        SARs = SARli(lim=n)
        for cnt, tick in gettick():
            t, state, act, r1, terminal = tick
            state = encState(state)
            SARs.update(t-1, state, act, r1, terminal)
            if len(SARs) == n:
                break
        idx = sample(range(n), n/2)
        ret = SARs[idx]
        agent.sa_sample = ret

        for cnt, tick in gettick():
            perf = callupdate(tick)

            if len(agent.SARs) < N_REPSIZE:
                continue
            if cnt % 100 == 0:
                print perf
            if cnt >= 10000:
                break
        print perf

        if saveflag:
            perfdic[idx] = float(perf)
            agent.save(idx)
            ysave(perfdic, 'perfdic.yaml')
    except:
        print_exc()
        set_trace()


def test():
    elems = tf.Variable(np.arange(10, dtype=np.float32))
    elems = tf.identity(elems)
    op_sum = tf.foldl(lambda a, x: a + x, elems)
    op_mean = tf.reduce_mean(elems)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print sess.run([op_sum, op_mean])


def fdebug(self):
    rets = []
    ret2 = []
    for sa0 in self.sa_sample:
        li, li2 = [], []
        for t in xrange(0, len(sa0), N_BATCH):
            try:
                S, A, R, S1, terminals = sa0[t:(t+N_BATCH)]
                loss_i, loss, Qsa = self.sess.run(
                    [self.loss_i, self.loss, self.Qsa],
                    feed_dict={
                        self.state: S,
                        self.act: A,
                        self.r: R,
                        self.state1: S1,
                        }
                    )
                li.append(np.mean(loss))
                li2.append(np.mean(np.power(R, 2)))
            except:
                print_exc()
                set_trace()
        rets.append(round(np.mean(li), 4))
        ret2.append(round(np.mean(li2), 4))
    ret = np.c_[rets, ret2]
    ret = np.c_[ret, ret[:, 0]/ret[:, 1]]
    print ret


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file')
    args_raw = parser.parse_args()
    args = vars(args_raw)
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
