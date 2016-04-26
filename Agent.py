#!/usr/bin/env python
# -*- coding: utf-8 -*-

from traceback import print_exc
import itertools as it
import tensorflow as tf
import numpy as np
from pdb import set_trace
from random import random, randint
from utils import chkEmpty, StateAct, encState
import savedata


# Implementation of Neural-Q
# Use tensorflow to construct neural network framework
# Need to define reward, loss, training process
# Unlike usual NN, the training and test process do at the same time.
# When we try to predict an action, we update our network when new state/reward arrives.
SEED = 34654
N_BATCH = 10
N_REPSIZE = 200


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
    def __init__(self, alpha=0.5, gamma=0.5, epsilon=0.1, **kwargs):
        algo = 'ANN'
        self.SARs = []  # List of (state, action)
        self.alpha = kwargs.get('alpha', 0.5)
        self.gamma = kwargs.get('gamma', 0.5)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.nRun = kwargs.get('nRun', 100)
        self.nolearn = kwargs.get('nolearn', False)
        self.algo = algo
        self.RepSize = N_REPSIZE

        self.Q = tf.placeholder(tf.float32, shape=[N_BATCH, 4])
        dic = eval(algo)(N_BATCH)  # Build up NN structure
        self.__dict__.update(dic)

        f = lambda x: tf.nn.l2_loss(self.__getattribute__(x))
        loss = tf.reduce_mean(tf.square(self.model - self.Q))
        regularizer = sum(map(f, self.parms))
        self.loss = loss #+ 1e-4 * regularizer
        self.optimizer = \
            tf.train.GradientDescentOptimizer(0.5)\
              .minimize(self.loss)

        # Before starting, initialize the variables.  We will 'run' this first.
        self.init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess = tf.Session()
        self.sess.run(self.init)

        if not self.nolearn:
            self.saveobj = savedata.SaveObj(
                self.algo + '.h5',
                [(p, self.__getattribute__(p).get_shape().as_list())
                    for p in self.parms],
                times=self.nRun,
                )

    def reset(self):
        assert self.SARs[-1].state1 is not None
        if len(self.SARs) > self.RepSize:
            self.SARs = self.SARs[N_BATCH:]

    def update(self, state, r0):
        state = encState(state)
        # receive state class
        if self.SARs and (not self.nolearn):
            s0 = self.SARs[-1]
            if s0.state1 is None:
                s0.state1 = state
                s0.score = r0
                self._update([s0])
        return self

    def booking(self, SA):
        self.SARs.append(SA)

    def _update(self, SARs):
        try:
            S = np.vstack([sa.state for sa in SARs])
            n1 = S.shape[0]
            S1 = np.vstack([sa.state1 for sa in SARs])
            r0 = np.vstack([self.reward(sa.act, sa.r()) for sa in SARs])
            if n1 < N_BATCH:
                S = np.r_[S, self.zeros(N_BATCH-n1)]
                S1 = np.r_[S1, self.zeros(N_BATCH-n1)]
                r0 = np.r_[r0, np.zeros((N_BATCH-n1, 4))]

            r01 = self.maxR(S1) * self.gamma
            for i, sa in enumerate(SARs):
                r0[i, sa.act] += r01[i]

            R = (1 - self.alpha) * self.eval(S) + self.alpha * r0
            feed_dict = {self.state: S, self.Q: R}
            var_list = [self.optimizer, self.loss]
            _, l = self.sess.run(var_list, feed_dict)
        except:
            print_exc()
            set_trace()

    def predict(self, state):
        state = encState(state)
        """ epsilon-greedy algorithm """
        if random() > self.epsilon:
            return self._action(state)
        else:
            return rndAction(state)

    def _action(self, state):
        """
        Return best action given state and win/loss
        Like predict process in NN
        Best action: argmax( R(s, a) + gamma * max(R(s', a')) )
        """
        try:
            act = -1
            rewards = self.eval(state)
            for i in np.argsort(rewards[0, :].ravel())[::-1]:
                act = i
                break
            assert act != -1
            self.booking(StateAct(state, act, None))
            return act
        except:
            print_exc()
            set_trace()

    def replay(self):
        # R(t+1) = a * R'(St, At) + (1-a) * (R(St, At) + g * max_a(R'(St1, a)))
        N = len(self.SARs)
        if (N < N_BATCH) or (not self.nolearn):
            return
        idx = np.random.choice(range(N), N_BATCH)
        # idx = np.array(range(N_BATCH))
        SARs = [self.SARs[i] for i in idx]
        self._update(SARs)

    def saveNN(self):
        if self.nolearn:
            return
        self.saveobj.save(self.parms, self.getparm())

    def reward(self, a, r):
        """
        Reward function
        """
        rmat = np.zeros([4], dtype=np.float32)
        if r != 0:
            rmat[a] = r
        return rmat

    def rewardS(self, sa):
        return self.reward(sa.act, sa.score)

    def maxR(self, state):
        return self.eval(state).max(axis=1)

    def eval(self, state):
        assert type(state) == np.ndarray, type(state)
        n = state.shape[0]
        if n < N_BATCH:
            state = np.r_[state, self.zeros(N_BATCH-n)]
        try:
            r, = self.sess.run(
                [self.model],
                feed_dict={self.state: state})
            return r
        except:
            print_exc()
            set_trace()

    def getparm(self):
        li = []
        for parm in it.imap(self.__getattribute__, self.parms):
            li.append(self.sess.run(parm))
        return li

    def load(self):
        """ TODO """


def ANN(N_BATCH):
    zeros = lambda x: np.zeros((x, 4, 4, 1))
    new_shape = (N_BATCH, 4, 4, 1)
    state = tf.placeholder(tf.float32, shape=new_shape)
    newstate = tf.reshape(state, [N_BATCH, 4 * 4])
    fc1_weights = tf.Variable(
        tf.truncated_normal([16, 32], stddev=0.1, seed=SEED)
        )
    fc1_biases = tf.Variable(
        tf.zeros([32]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([32, 4], stddev=0.1, seed=SEED)
        )
    fc2_biases = tf.Variable(
        tf.zeros([4]))
    parms = ('fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases')

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
        tf.truncated_normal([2, 2, 1, 16], stddev=0.1, seed=SEED)
        )
    conv1_biases = tf.Variable(
        tf.zeros([16]))
    fc1_weights = tf.Variable(
        tf.truncated_normal([256, 4], stddev=0.1, seed=SEED)
        # tf.ones([N, ncol])
        )
    fc1_biases = tf.Variable(
        tf.zeros([4]))
    parms = ('conv1_weights', 'conv1_biases', 'fc1_weights', 'fc1_biases')

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


def CNN2(self):
    self.zeros = lambda x: np.zeros((x, 6, 7, 2))
    self.new_shape = (N_BATCH, 6, 7, 2)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)

    self.conv1_weights = tf.Variable(
        tf.truncated_normal([3, 3, 2, 16], stddev=0.1, seed=SEED)
        )
    self.conv1_biases = tf.Variable(
        tf.zeros([16]))
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([672, 7], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('conv1_weights', 'conv1_biases', 'fc1_weights', 'fc1_biases')

    conv = tf.nn.conv2d(
        self.state,
        self.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 1, 1],
        padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]
        )

    self.model = tf.nn.softmax(
        tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)


def CNN3(self):
    self.new_shape = (1, 6, 7, 1)
    self.state = tf.placeholder(tf.float32, shape=self.new_shape)

    self.conv1_weights = tf.Variable(
        tf.truncated_normal([4, 4, 1, 32], stddev=0.1, seed=SEED)
        )
    self.conv1_biases = tf.Variable(
        tf.zeros([32]))
    self.conv2_weights = tf.Variable(
        tf.truncated_normal([3, 4, 32, 64], stddev=0.1, seed=SEED)
        )
    self.conv2_biases = tf.Variable(
        tf.zeros([64]))
    self.fc1_weights = tf.Variable(
        tf.truncated_normal([256, 512], stddev=0.1, seed=SEED)
        )
    self.fc1_biases = tf.Variable(
        tf.zeros([512]))
    self.fc2_weights = tf.Variable(
        tf.truncated_normal([512, 7], stddev=0.1, seed=SEED)
        )
    self.fc2_biases = tf.Variable(
        tf.zeros([7]))
    self.parms = ('conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases')

    conv = tf.nn.conv2d(
        self.state,
        self.conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv = tf.nn.conv2d(
        pool,
        self.conv2_weights,
        strides=[1, 1, 1, 1],
        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
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
        tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
    model = tf.nn.softmax(
        tf.matmul(hidden, self.fc2_weights) + self.fc2_biases)
    self.model = model


