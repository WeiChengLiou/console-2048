#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from abc import types
SEED = 34654
N_BATCH = 100
N_REPSIZE = 500


def rndAction(state):
    try:
        return randint(0, 3)
    except:
        print_exc()
        set_trace()


class SARli(types.ListType):
    def state(self):
        return np.vstack([x.state for x in self])

    def act(self):
        return np.vstack([x.act for x in self])

    def r(self):
        return np.vstack([x.r() for x in self])

    def state1(self):
        return np.vstack([x.state1 for x in self])


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

    def update(self, s1, r):
        return self

    def reset(self):
        """"""


class NNQ(Model):
    def __init__(self, **kwargs):
        algo = kwargs.get('algo', 'ANN')
        print('Use %s' % algo)
        self.SARs = SARli()  # List of (state, action)
        self.alpha = kwargs.get('alpha', 0.5)
        self.gamma = kwargs.get('gamma', 0.5)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.nRun = kwargs.get('n', 100)
        self.nolearn = not kwargs.get('train', True)
        self.kw = kwargs.get('kw', '')
        self.savegame = kwargs.get('savegame', False)
        self.trainNFQ = kwargs.get('trainNFQ', False)
        self.algo = algo
        self.RepSize = N_REPSIZE
        self.score = 0
        self.i0 = 0

        self.state = tf.placeholder(tf.float32, shape=(N_BATCH, 4, 4, 4))
        self.acts = tf.placeholder(tf.int32, shape=[N_BATCH])
        self.r = tf.placeholder(tf.float32, shape=[N_BATCH, 1])
        self.state1 = tf.placeholder(tf.float32, shape=(N_BATCH, 4, 4, 4))

        self.model_init = eval(algo)
        self.model = self.model_init(self.state, 'model', reuse=None)
        self.Qhat = self.model_init(self.state1, 'model', reuse=True)
        self.loss_def(
            self.acts,
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
        tf.scalar_summary(
            'Target.loss', self.TargetNetwork_opt.loss)
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

        # if not self.nolearn:
        #     self.saveobj = savedata.SaveObj(
        #         self.algo + '.h5',
        #         [(p.name, p.get_shape().as_list()) for p in self.parms],
        #         times=self.nRun,
        #         )

    def show(self):
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

    def update(self, state1, r0):
        # receive state class
        if (len(self.records) <= 3):
            return self

        if self.SARs and (not self.nolearn):
            if not self.trainNFQ:
                state1 = self.encState(state1)
            s0 = self.SARs[-1]
            if (s0.state1 is None) or self.trainNFQ:
                s0.state1 = state1
                s0.score = encReward(r0)
        return self

    def booking(self, state, act):
        try:
            if self.SARs:
                num0 = hash(tuple(state.ravel()))
                num1 = hash(tuple(self.SARs[-1].state.ravel()))
                if num0 == num1:
                    s0 = self.SARs[-1]
                    s0.act = act
                    return
            assert isinstance(state, np.ndarray)
            self.SARs.append(StateAct(state, act, None))
        except:
            print_exc()
            set_trace()

    def _update(self, Target=False):
        if self.nolearn:
            return
        N = len(self.SARs)
        if (N < N_REPSIZE) or (self.nolearn):
            return None

        iter1 = self.ExpReplay_opt.getli(self.SARs)
        # SARs = it.islice(iter1, 1)
        for t, SARs in enumerate(iter1):
            S = SARs.state()
            A = SARs.act().ravel()
            R = SARs.r()
            S1 = SARs.state1()

            parms = self.sess, S, A, R, S1, self.summary
            if Target:
                ret = self.TargetNetwork_opt.optimize(parms)
            else:
                ret = self.optimize(*parms)

            if ret is not None:
                loss_res, merge_res = ret
                if t and (t % 100 == 0):
                    self.writer_sum.add_summary(
                        merge_res, self.tickcnt)
                    self.tickcnt += 1

    def predict(self, state):
        # state = encState(state)
        """ epsilon-greedy algorithm """
        if (len(self.records) < 3):
            return rndAction(state)

        state = self.encState(state)
        if random() < self.epsilon:
            act = rndAction(state)
        else:
            act = self._action(state)
        if not self.trainNFQ:
            self.booking(state, act)
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

        for i in np.argsort(rewards[0, :].ravel())[::-1]:
            act = i
            break
        assert act != -1
        return act

    def getparm(self):
        return [(p.name, val) for p, val in
                zip(self.parms, self.sess.run(self.parms))]

    def reset(self):
        self.score = 0
        if not self.nolearn:
            assert self.SARs[-1].state1 is not None
        if len(self.SARs) > self.RepSize:
            self.SARs = self.SARs[N_BATCH:]
        self.i0 = len(self.SARs)

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

    def loss_def(self, acts, r, gamma):
        Qsa = fgetidx(self.model, self.acts)
        self.loss = r + gamma * maxQ(self.Qhat) - Qsa
        self.op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def optimize(self, sess, state, acts, r, state1, summary=None):
        varlist = [self.loss, self.op]
        if summary is not None:
            varlist.append(summary)

        ret = sess.run(
            varlist,
            feed_dict={
                self.state: s0,
                self.acts: acts,
                self.r: r,
                self.s1: s1
                })

        if summary is not None:
            return ret[0], ret[2]
        else:
            return ret[0]


def ANN(state, layer='', reuse=None):
    with tf.variable_scope(layer, reuse=reuse):
        model = relu(full_layer(state, 256), layer='layer1', reuse=reuse)
        model = relu(full_layer(model, 64), layer='layer2', reuse=reuse)
        model = relu(full_layer(model, 16), layer='layer3', reuse=reuse)
        model = relu(full_layer(model, 4), layer='layer4', reuse=reuse)
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


def getidx(mat, acts):
    # Get element value by index with each row
    nrow, ncol = mat.get_shape().as_list()
    mat_1d = tf.reshape(mat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, acts)
    ret = tf.gather(mat_1d, idx)
    return ret


fgetidx = tf.make_template('getidx', getidx)


def maxQ(Q):
    Q_max = tf.reduce_max(Q, reduction_indices=[1])
    size0 = getshape(Q_max)
    return tf.reshape(Q_max, [size0[0], 1])


class ExpReplay(object):
    def __init__(self, obj, shuffle=False):
        self.loss = obj.loss
        self.op = obj.op

    def optimize(self, sess, state, acts, r, state1, summary=None):
        varlist = [self.loss, self.op]
        if summary is not None:
            varlist.append(summary)

        ret = sess.run(
            varlist,
            feed_dict={
                self.state: s0,
                self.acts: acts,
                self.r: r,
                self.s1: s1
                })

        if summary is not None:
            return ret[0], ret[2]
        else:
            return ret[0]

    def getli(self, SARs_raw):
        N = len(SARs_raw)
        idx = np.random.permutation(range(N))
        for t in xrange(0, N, N_BATCH):
            SARs = [SARs_raw[i] for i in idx[t:(t+N_BATCH)]]
            if len(SARs) != N_BATCH:
                continue
            assert len(SARs) == N_BATCH, (t, map(len, (SARs, self.SARs)))
            yield SARli(SARs)


class TargetNetwork(object):
    """
    - Use old weight to calculate:
        Q0(r, gamma, s1, Q_0) = r + gamma * max_a Q_0(s1, a)
    - Loss Tensor:
        Loss(Q0, s0, acts, Q) = mean(sqrt(Q0 - Q(s0, acts)))
    - Minimize Loss:
        Optim(Q0, s0, acts, Q) = tf.train.AnyOptimizer(Loss)
    """
    def __init__(self, obj, enable=False):
        self.state = obj.state
        self.acts = obj.acts
        self.r = obj.r
        self.enable = enable

        # initialize another NN structure
        self.model_old = obj.model_init(self.state, 'model0')
        self.assigns = setAssign(obj.model, self.model_old)
        self.firstRun = False

        nrow, ncol = getshape(self.model_old)
        self.Qval_old = tf.placeholder(
            self.model_old.dtype,
            shape=[nrow, 1],
            )
        self.Qhat_old = self.r + obj.gamma * maxQ(self.model_old)
        Qsa = fgetidx(obj.model, self.acts)
        self.loss = tf.reduce_mean(
            tf.square(self.Qval_old - Qsa)
            )
        self.op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

    def optimize(self, sess, state, acts, r, state1, summary=None):
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
                self.state: state1,
                })

        varlist = [self.loss, self.op]
        if summary is not None:
            varlist.append(summary)

        ret = sess.run(
            varlist,
            feed_dict={
                self.Qval_old: Qhat,
                self.state: state,
                self.acts: acts,
                }
            )

        sess.run(self.assigns)

        if summary is not None:
            return ret[0], ret[2]
        else:
            return ret[0]


def NFQ(**kwargs):
    saveflag = True
    agent = NNQ(**kwargs)
    # agent.load()
    agent.listparm()

    SARs = agent.SARs
    nlim = 20000
    train = range(188)
    test = range(188, 363)
    ALL = range(nlim)
    flag = train
    gets = lambda rs: list(it.chain.from_iterable([r[0] for r in rs]))

    for i, fi in enumerate(os.listdir('data')):
        if i not in flag:
            continue
        fi = 'data/%s' % fi
        rets = np.load(fi)['arr_0']
        agent.records = []
        rec = agent.records
        jend = len(rets) - 1

        for j, ret in enumerate(rets):
            # print j, ret[2]
            rec.append(ret)
            if len(rec) >= 5:
                r = ret[2]
                if r >= 0:
                    r += 2
                s1 = agent.encState(gets(rec[-4:]), noadd=True)
                agent.update(s1, r)
            if (len(rec) >= 4) and (j != jend):
                s0 = agent.encState(gets(rec[-4:]), noadd=True)
                act = ret[1]
                agent.booking(s0, act)

        for t, sa in enumerate(agent.SARs):
            assert sa.state1 is not None, t

        if i % 5 == 0:
            loss = agent._update()
            if loss is not None:
                print i, np.mean(loss), len(SARs)
        agent.reset()

    # loss = agent.replay()
    # print i, np.mean(loss), len(SARs)
    if saveflag:
        agent.save()


def test():
    tf.reset_default_graph()
    state = tf.placeholder(tf.float32, shape=(N_BATCH, 4, 4, 4))
    acts = tf.placeholder(tf.int32, shape=[N_BATCH])
    r = tf.placeholder(tf.float32, shape=[N_BATCH, 1])
    state1 = tf.placeholder(tf.float32, shape=(N_BATCH, 4, 4, 4))
    model = relu(conv_layer(state, [2, 2], 16, 1))
    model = relu(conv_layer(model, [2, 2], 32, 1))
    globals().update(locals())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file')
    args = parser.parse_args()
    args = vars(args)
    if args['config']:
        config = yaml.load(open(args['config'], 'rb'))
        args.update(config)
    args['agent'] = args['agent'].lower()
    print(args)

    if args.get('N_BATCH'):
        N_BATCH = args['N_BATCH']
    if args.get('N_REPSIZE'):
        N_REPSIZE = args['N_REPSIZE']

    NFQ(**args)
