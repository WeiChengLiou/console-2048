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
        self.SARs = []  # List of (state, action)
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

        self.loss, self.optimizer = RL_LossFunc(
            self.parms, self.model, self.acts, self.Q)
        self.saver = tf.train.Saver(self.parms)

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(self.init)

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
        S = np.vstack([sa.state for sa in SARs])
        S1 = np.vstack([sa.state1 for sa in SARs])
        A = np.vstack([sa.act for sa in SARs])
        R = np.vstack([self.reward(sa.act, sa.r()) for sa in SARs])

        r1 = self.gamma * self.maxR(S1)
        for i, sa in enumerate(SARs):
            R[i, sa.act] += r1[i]

        feed_dict = {self.state: S, self.Q: R, self.acts: A.ravel()}
        var_list = [self.optimizer, self.loss]
        _, losshat = self.sess.run(var_list, feed_dict)
        return losshat

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

        li = []
        idx = np.random.permutation(range(N))
        try:
            for t in xrange(30000):
                SARs = [self.SARs[i] for i in idx[:N_BATCH]]
                assert len(SARs) == N_BATCH, len(SARs)
                ret = self._update(SARs)
                li.append(ret)
                if (t >= 1000) and (t % 1000 == 0):
                    print t, ret

                idx = idx[N_BATCH:]
                if len(idx) <= N_BATCH:
                    idx = np.random.permutation(range(N))
        except:
            print_exc()
            set_trace()
        li = np.convolve(np.array(li), np.ones(10)/10, mode='full')[:-10]
        plt.plot(li[100::100][1:])
        plt.show()
        return ret

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


def getidx(smat, acts):
    # Get element value by index with each row
    nrow, ncol = smat.get_shape().as_list()
    smat1d = tf.reshape(smat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, acts)
    ret = tf.gather(smat1d, idx)
    return ret


def feval(model, s, a):
    return getidx(model, a)


def ExpReplay(model, s_, a_, r_, s1_, gamma):
    Q = getidx(model, a)
    Q1 = feval(model, s1_, a_)
    return r_ + gamma * Q1 - Q


def RL_LossFunc(parms, mat, acts, target):
    sret = getidx(mat, acts)
    tret = getidx(target, acts)
    loss = tf.reduce_mean(tf.square(
        tf.sub(sret, tret)
        ))
    regularizer = sum(map(tf.nn.l2_loss, parms))
    loss += 1e-4 * regularizer
    optim_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
    return loss, optim_op


def NFQ(**kwargs):
    saveflag = True
    agent = NNQ(**kwargs)
    agent.load()
    agent.listparm()

    SARs = agent.SARs
    nlim = 20000
    train = range(64)
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
        loss = agent.replay()
        if loss is not None:
            print i, np.mean(loss), len(SARs)
            break
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
