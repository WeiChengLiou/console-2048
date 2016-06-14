#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
from pdb import set_trace


class SARli(object):
    def __init__(self):
        self.states = []
        self.acts = []
        self.rs = []
        self.terminals = []
        self.idx = -1
        self.game_t = -1

    def state(self, idx=None):
        if idx is None:
            idx = self.idx
        assert idx <= len(self.states)
        return self.states[idx]

    def act(self, idx=None):
        if idx is None:
            idx = self.idx
        assert idx <= len(self.acts)
        return self.acts[idx]

    def r(self, idx=None):
        if idx is None:
            idx = self.idx
        assert idx <= len(self.rs)
        return self.rs[idx]

    def update(self, t, state, act, r, terminal):
        state.flags.writeable = False
        self.states.append(state)
        self.acts.append(act)
        self.rs.append(r)
        self.terminals.append(terminal)
        self.idx += 1
        self.game_t = t

    def eq_np(self, x1, x2):
        return hash(x1.data) == hash(x2.data)

    def fixSA(self, t, state, act):
        print 'fixSA maybe deprecated'
        state.flags.writeable = False
        assert self.eq_np(self.states[t], state), 'Update wring state'
        self.acts[t] = act
