#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pdb


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


vfunc = np.vectorize(lambda x, y: np.log2(max(x, y)))


def encState(state):
    """ encode original state into two boards """
    s1 = np.array(state).astype(np.float)
    s1 = vfunc(s1, 1)
    return s1.reshape((1, 4, 4, 1))


def chkEmpty(s1, i):
    assert s1.shape[0] == 1, s1.shape
    c = i % 7
    r = (i - c) / 7
    return (s1[0, r, c, 0] == 0) and (s1[0, r, c, 1] == 0)


