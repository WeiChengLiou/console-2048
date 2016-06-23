#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import yaml


class State(object):
    # State Class

    def __init__(self, state, win):
        self.state = state
        self.win = win

    def state(self):
        return self.state


class StateAct(object):
    def __init__(self, state, act, score):
        self.state = state
        self.act = act
        self.score = score
        self.state1 = None

    def r(self):
        if self.score is None:
            return 0.
        else:
            return self.score


zeros = lambda x: np.zeros((x, 4, 4, 1))


def encState(state):
    """ encode original state as [1, row, col]"""
    for i in range(4):
        assert len(state[i]) == 4
    s1 = np.array(state).astype(np.float)
    return np.log2(s1 + 1).reshape((4, 4, 1))


def chkEmpty(s1, i):
    assert s1.shape[0] == 1, s1.shape
    c = i % 7
    r = (i - c) / 7
    return (s1[0, r, c, 0] == 0) and (s1[0, r, c, 1] == 0)


def encReward(r):
    return np.log2(r + 1)


def ysave(obj, fi):
    yaml.dump(obj, open(fi, 'wb'))


def yload(fi):
    return yaml.load(open(fi, 'rb'))

