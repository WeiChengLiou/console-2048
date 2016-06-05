#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pylab as plt
import copy
from tftools import *
import os
import yaml
from datetime import datetime
from traceback import print_exc
import itertools as it
import tensorflow as tf
import numpy as np
from pdb import set_trace
from random import random, randint
from utils import chkEmpty, StateAct, encState, encReward
import savedata
from operator import mul


# Implementation of Neural-Q
# Use tensorflow to construct neural network framework
# Need to define reward, loss, training process
# Unlike usual NN, the training and test process do at the same time.
# When we try to predict an action, we update our network when new state/reward arrives.
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
        return [x.state for x in self]

    def act(self):
        return [x.act for x in self]

    def r(self):
        return [x.r() for x in self]

    def state1(self):
        return [x.state1 for x in self]


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

        self.Q = tf.placeholder(tf.float32, shape=[N_BATCH, 4])
        self.__dict__.update(
            eval(algo)(N_BATCH)  # Build up NN structure
            )
        self.parms = tf.trainable_variables()
        self.acts = tf.placeholder(tf.int32, shape=[N_BATCH])

        for x in self.parms:
            tf.histogram_summary(x.name, x)

        self.loss, self.optimizer = RL_LossFunc(
            self.parms, self.model, self.acts, self.Q)
        self.saver = tf.train.Saver(self.parms)

        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.writer_sum = tf.train.SummaryWriter(
            'tmp/logs', self.sess.graph_def)
        self.tickcnt = 0

        if not self.nolearn:
            self.saveobj = savedata.SaveObj(
                self.algo + '.h5',
                [(p.name, p.get_shape().as_list()) for p in self.parms],
                times=self.nRun,
                )

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
                # self._update([s0])
                # self.replay()
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

    def _update(self, SARs):
        """
        Q-learning:
            R(t+1) = r + gamma * max_a(R'(St1, a)) - R(St, At)
            https://en.wikipedia.org/wiki/Q-learning
        """
        S = np.vstack(SARs.state)
        S1 = np.vstack(SARs.state1)
        A = np.vstack(SARs.act)
        R = np.vstack([self.reward(sa.act, sa.r()) for sa in SARs])

        r1 = self.gamma * self.maxR(S1)
        for i, sa in enumerate(SARs):
            R[i, sa.act] += r1[i]

        feed_dict = {self.state: S, self.Q: R, self.acts: A.ravel()}
        var_list = [self.optimizer, self.loss, self.summary]
        _, loss_res, merge_res = self.sess.run(var_list, feed_dict)
        return loss_res, merge_res

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
        rewards = self.eval(state)
        for i in np.argsort(rewards[0, :].ravel())[::-1]:
            act = i
            break
        assert act != -1
        return act

    def reward(self, a, r):
        """
        Reward function
        """
        rmat = np.zeros([4], dtype=np.float32)
        if r != 0:
            rmat[a] = r
        return rmat

    def rewardS(self, sa):
        return self.reward(sa.act, sa.r)

    def maxR(self, state):
        return self.eval(state).max(axis=1)

    def eval(self, state):
        assert type(state) == np.ndarray, type(state)
        state = self.FULL(state)
        r, = self.sess.run(
            [self.model],
            feed_dict={self.state: state})
        return r

    def getparm(self):
        return [(p.name, val) for p, val in
                zip(self.parms, self.sess.run(self.parms))]

    def subreplay(self):
        r = 0
        # for sa in self.SARs[self.i0:][::-1]:
        #     sa.score += r * self.gamma
        #     r = sa.score
            # if r:
            #     set_trace()

    def replay(self):
        self.subreplay()
        if self.nolearn:
            return
        N = len(self.SARs)
        if (N < N_REPSIZE) or (self.nolearn):
            return None

        idx = np.random.permutation(range(N))
        for t in xrange(0, N_REPSIZE, N_BATCH):
            SARs = SARli([self.SARs[i] for i in idx[t:(t+N_BATCH)]])
            if len(SARs) != N_BATCH:
                continue
            assert len(SARs) == N_BATCH, (t, map(len, (SARs, self.SARs)))
            loss_res, merge_res = self._update(SARs)
            if t and (t % 100 == 0):
                self.writer_sum.add_summary(
                    merge_res, self.tickcnt)
                self.tickcnt += 1
        return loss_res

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
        self.saveobj.save(self.getparm())
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
        return np.vstack([s0, self.zeros(n)])

    def encState(self, state, noadd=False):
        s1 = []
        if not noadd:
            [s1.extend(copy.deepcopy(r[0])) for r in self.records[-3:]]
        s1.extend(copy.deepcopy(state))
        assert len(s1) == 16
        return encState(s1)


def ANN(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 4))
    new_shape = (N_BATCH, 4, 4, 4)
    state = tf.placeholder(tf.float32, shape=new_shape)
    normfun = lambda size: tf.truncated_normal(size, stddev=0.1, seed=SEED)

    model = relu(add_fullcon(state, 256))
    model = relu(add_fullcon(model, 64))
    model = relu(add_fullcon(model, 16))
    model = relu(add_fullcon(model, 4))
    return locals()


def CNN(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 4))
    new_shape = (N_BATCH, 4, 4, 4)
    state = tf.placeholder(tf.float32, shape=new_shape)
    normfun = lambda size: tf.truncated_normal(size, stddev=0.1, seed=SEED)

    model = relu(add_conv(state, [2, 2], 16, 1))
    model = relu(add_conv(model, [2, 2], 32, 1))
    model = relu(add_fullcon(model, 256))
    model = relu(add_fullcon(model, 4))
    return locals()


def getidx(mat, acts):
    # Get element value by index with each row
    nrow, ncol = mat.get_shape().as_list()
    mat_1d = tf.reshape(mat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, acts)
    ret = tf.gather(mat_1d, idx)
    return ret


fgetidx = tf.make_template('getidx', getidx)


def maxR(fQ, state1):
    assert isinstance(state1, tf.python.framework.ops.Tensor)
    R1_mat = fQ(state1)
    R1_max = tf.reduce_max(R1_mat, reduction_indices=[1])
    size0 = getshape(R1_max)
    size1 = [size0[0], 1]
    return tf.reshape(R1_max, size1)


def Target_LossFunc(obj, s0, acts, r, s1):
    """
    - Use old weight to calculate:
        R1(r, gamma, s1, Q) = r + gamma * max_a Q(s1, a)
    - Loss Tensor:
        Loss(R1, s0, acts, Q) = mean(sqrt(R1 - Q(s0, acts)))
    - Minimize Loss:
        Optim(R1, s0, acts, Q) = tf.train.AnyOptimizer(Loss)
    """
    R1 = maxR(obj.fQ, s1)
    target = r + obj.gamma * R1_max


def ExpReplay_LossFunc(model):
    """
    - Define Loss:
        Loss(r, gamma, s1, s0, acts, Q) =
            mean(sqrt(r + gamma * max_a Q(s1, a) - Q(s0, acts)))
    - Minimize Loss:
        Optim(r, gamma, s1, s0, acts, Q) = tf.train.AnyOptimizer(Loss)
    """


def RL_LossFunc(parms, model, acts, Q):
    sret = fgetidx(model, acts)
    tret = fgetidx(Q, acts)
    loss = tf.reduce_mean(tf.square(
        tf.sub(sret, tret)
        ))
    regularizer = sum(map(tf.nn.l2_loss, parms))
    loss += 1e-4 * regularizer
    optim_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    return loss, optim_op


def NFQ(**kwargs):
    saveflag = True
    agent = NNQ(**kwargs)
    agent.load()
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
            loss = agent.replay()
            if loss is not None:
                print i, np.mean(loss), len(SARs)
        agent.reset()

    # loss = agent.replay()
    # print i, np.mean(loss), len(SARs)
    if saveflag:
        agent.save()


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
