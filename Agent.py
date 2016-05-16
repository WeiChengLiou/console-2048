#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from traceback import print_exc
import itertools as it
import tensorflow as tf
import numpy as np
from pdb import set_trace
from random import random, randint
from utils import chkEmpty, StateAct, encState, encReward
import savedata


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

    def update(self, state, r0):
        score0, self.score = self.score, r0
        state = encState(state)
        # receive state class
        if self.SARs and (not self.nolearn):
            s0 = self.SARs[-1]
            if s0.state1 is None:
                s0.state1 = state
                s0.score = encReward(r0 - score0)
                # self._update([s0])
        return self

    def booking(self, state, act):
        if self.SARs:
            num0 = hash(tuple(state.ravel()))
            num1 = hash(tuple(self.SARs[-1].state.ravel()))
            if num0 == num1:
                s0 = self.SARs[-1]
                s0.act = act
                return
        s0 = StateAct(state, act, None)
        self.SARs.append(s0)

    def _update(self, SARs):
        """
        Q-learning:
            R(t+1) = a * R'(St, At) +
                     (1-a) * (R(St, At) + g * max_a(R'(St1, a)))
            https://en.wikipedia.org/wiki/Q-learning
        """
        S = np.vstack([sa.state for sa in SARs])
        n1 = S.shape[0]
        S1 = np.vstack([sa.state1 for sa in SARs])
        A = np.vstack([sa.act for sa in SARs])
        r0 = np.vstack([self.reward(sa.act, sa.r()) for sa in SARs])
        if n1 < N_BATCH:
            dN = N_BATCH - n1
            S = np.r_[S, self.zeros(dN)]
            S1 = np.r_[S1, self.zeros(dN)]
            A = np.r_[S, self.zeros(dN)]
            r0 = np.r_[r0, np.zeros((dN, 4))]

        r01 = self.maxR(S1) * self.gamma
        for i, sa in enumerate(SARs):
            r0[i, sa.act] += r01[i]

        R = (1 - self.alpha) * self.eval(S) + self.alpha * r0
        feed_dict = {self.state: S, self.Q: R, self.acts: A.ravel()}
        var_list = [self.optimizer, self.loss]
        _, l = self.sess.run(var_list, feed_dict)

    def predict(self, state):
        state = encState(state)
        """ epsilon-greedy algorithm """
        if random() > self.epsilon:
            act = self._action(state)
        else:
            act = rndAction(state)
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

    def subreplay(self):
        r = 0
        li = [(sa.state, sa.act, sa.state1) for sa
              in self.SARs[self.i0:][::-1]]
        np.savez_compressed(
            datetime.now().strftime('data/game.%Y%m%d%H%M%S.npz'),
            li)
        for sa in self.SARs[self.i0:][::-1]:
            # sa.score += r * self.gamma
            r = sa.score

    def replay(self):
        self.subreplay()
        if self.nolearn:
            return
        N = len(self.SARs)
        if (N < N_BATCH) or (self.nolearn):
            print N, 'NO Replay'
            return
        idx = np.random.choice(range(N), N_BATCH)
        # idx = np.array(range(N_BATCH))
        SARs = [self.SARs[i] for i in idx]
        self._update(SARs)

    def reset(self):
        self.score = 0
        assert self.SARs[-1].state1 is not None
        if len(self.SARs) > self.RepSize:
            self.SARs = self.SARs[N_BATCH:]
        self.i0 = len(self.SARs)

    def save(self):
        if self.nolearn:
            return
        self.saveobj.save(self.getparm())
        self.saver.save(self.sess, 'tmp/%s.ckpt' % self.algo)

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
        n = state.shape[0]
        if n < N_BATCH:
            state = np.r_[state, self.zeros(N_BATCH-n)]

        r, = self.sess.run(
            [self.model],
            feed_dict={self.state: state})
        return r

    def getparm(self):
        return [(p.name, val) for p, val in
                zip(self.parms, self.sess.run(self.parms))]

    def load(self):
        fi = 'tmp/%s.ckpt' % self.algo
        self.saver.restore(self.sess, fi)


def ANN(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 1))
    new_shape = (N_BATCH, 4, 4, 1)
    state = tf.placeholder(tf.float32, shape=new_shape)
    newstate = tf.reshape(state, [N_BATCH, 4 * 4])
    fc1_weights = tf.Variable(
        tf.truncated_normal([16, 32], stddev=0.1, seed=SEED),
        trainable=True,
        name='fc1_weights',
        )
    fc1_biases = tf.Variable(
        tf.zeros([32]),
        trainable=True,
        name='fc1_biases',
        )
    fc2_weights = tf.Variable(
        tf.truncated_normal([32, 4], stddev=0.1, seed=SEED),
        trainable=True,
        name='fc2_weights',
        )
    fc2_biases = tf.Variable(
        tf.zeros([4]),
        trainable=True,
        name='fc2_biases',
        )
    model = tf.nn.relu(
        tf.matmul(newstate, fc1_weights) + fc1_biases)
    model = tf.nn.softmax(
        tf.matmul(model, fc2_weights) + fc2_biases)
    return locals()


def CNN(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 1))
    new_shape = (N_BATCH, 4, 4, 1)
    state = tf.placeholder(tf.float32, shape=new_shape)

    conv1_weights = tf.Variable(
        tf.truncated_normal([2, 2, 1, 16], stddev=0.1, seed=SEED),
        trainable=True,
        name='conv1_weights',
        )
    conv1_biases = tf.Variable(
        tf.zeros([16]),
        trainable=True,
        name='conv1_biases',
        )
    fc1_weights = tf.Variable(
        tf.truncated_normal([256, 4], stddev=0.1, seed=SEED),
        trainable=True,
        name='fc1_weights',
        )
    fc1_biases = tf.Variable(
        tf.zeros([4]),
        trainable=True,
        name='fc1_biases',
        )

    conv = tf.nn.conv2d(
        state,
        conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    relu_shape = relu.get_shape().as_list()
    reshape = tf.reshape(
        relu,
        [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]]
        )
    model = tf.nn.softmax(
        tf.matmul(reshape, fc1_weights) + fc1_biases)
    return locals()


def CNN2(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 1))
    new_shape = (N_BATCH, 4, 4, 1)
    state = tf.placeholder(tf.float32, shape=new_shape)

    conv1_weights = tf.Variable(
        tf.truncated_normal([2, 2, 1, 16], stddev=0.1, seed=SEED),
        trainable=True,
        )
    conv1_biases = tf.Variable(
        tf.zeros([16]),
        trainable=True,
        )
    conv2_weights = tf.Variable(
        tf.truncated_normal([2, 2, 16, 32], stddev=0.1, seed=SEED),
        trainable=True,
        )
    conv2_biases = tf.Variable(
        tf.zeros([32]),
        trainable=True,
        )
    fc1_weights = tf.Variable(
        tf.truncated_normal([32, 64], stddev=0.1, seed=SEED),
        trainable=True,
        )
    fc1_biases = tf.Variable(
        tf.zeros([64]),
        trainable=True,
        )
    fc2_weights = tf.Variable(
        tf.truncated_normal([64, 4], stddev=0.1, seed=SEED),
        trainable=True,
        )
    fc2_biases = tf.Variable(
        tf.zeros([4]),
        trainable=True,
        )

    conv = tf.nn.conv2d(
        state,
        conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv = tf.nn.conv2d(
        pool,
        conv2_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )
    hidden = tf.nn.relu(
        tf.matmul(reshape, fc1_weights) + fc1_biases)
    model = tf.nn.softmax(
        tf.matmul(hidden, fc2_weights) + fc2_biases)
    return locals()


def get_idx(smat, acts):
    # Get element value by index with each row
    nrow, ncol = smat.get_shape().as_list()
    smat1d = tf.reshape(smat, [-1])
    rng = tf.constant(np.arange(nrow, dtype=np.int32) * ncol)
    idx = tf.add(rng, acts)
    ret = tf.gather(smat1d, idx)
    return ret


def RL_LossFunc(parms, mat, acts, target):
    sret = get_idx(mat, acts)
    tret = get_idx(target, acts)
    loss = tf.square(
        tf.sub(sret, tret)
        )
    # regularizer = sum(map(tf.nn.l2_loss, parms))
    # loss += 1e-4 * regularizer
    optim_op = tf.train.GradientDescentOptimizer(.1).minimize(loss)
    return loss, optim_op
