#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
from pdb import set_trace
from abc import types
from collections import namedtuple
from traceback import print_exc
SAR = namedtuple('SAR', ['state', 'act', 'r', 'terminal'])
s_zero = lambda: np.zeros((4, 4, 1), dtype=np.float32)


class SARli(types.ListType):
    def __init__(self, lim=100):
        self.sars = []
        self.idxs = []
        self.lim = lim

    def __len__(self):
        return len(self.idxs)
        if self.sars and (self.sars[-1].r is None):
            return len(self.idxs)
        else:
            return max(0, len(self.idxs) - 1)

    def hashable(self, s):
        assert isinstance(s, np.ndarray)
        s.flags.writeable = False
        return s

    def update(self, t, state, act, r, terminal, debug=False):
        try:
            if len(self.sars) == 0:
                assert t == 0
            state.flags.writeable = False
            sar = SAR(state, act, float(r), terminal)
            if self.sars:
                sa0 = self.sars[-1]
                if hash(sar.state.data) == hash(sa0.state.data):
                    return

            self.sars.append(sar)
            if t >= 3:
                self.idxs.append(len(self.sars)-1)
                if len(self.idxs) == 1:
                    assert len(self.sars) == 4
            if terminal:
                s0 = s_zero()
                s0.flags.writeable = False
                sar = SAR(s0, None, None, None)
                self.sars.append(sar)
            if len(self) > self.lim:
                idxs = self.idxs[1:]
                i0 = idxs[0] - 3
                idxs = [(j-idxs[0]+3) for j in idxs]
                sars = self.sars[i0:]
                assert all([(sars[j].act is not None) for j in idxs])
                self.idxs = idxs
                self.sars = sars
        except:
            print_exc()
            set_trace()

    def __getstate__(self, i):
        try:
            if len(self.sars) > i:
                li = [self.sars[j].state for j in range(i-0, i+1)]
            elif len(self.sars) == i:
                li = [self.sars[j].state for j in range(i-0, i)]
                li.append(s_zero())
            else:
                raise Exception('Index error')
            return np.vstack(li).reshape((1, 4, 4, 1))
        except Exception as e:
            raise Exception(e, i, len(self.sars), range(i-0, i+1))

    def __getitem__(self, i):
        if isinstance(i, list):
            return self._getbatch_(i)
        else:
            return self._getbatch_([i])

    def __getslice__(self, i0, i1):
        return self.__getitem__(range(i0, i1))

    def _getbatch_(self, idx):
        try:
            idx0 = [self.idxs[i] for i in idx]
            sars = [self.sars[i] for i in idx0]
            state = self.hashable(np.vstack(
                [self.__getstate__(i) for i in idx0]))
            act = np.array([s.act for s in sars])
            r = np.vstack([s.r for s in sars])
            terminal = np.array([s.terminal for s in sars])

            idx1 = [(self.idxs[i]+1) for i in idx]
            state1 = self.hashable(np.vstack(
                [self.__getstate__(i) for i in idx1]))
            assert all([(x is not None) for x in act])
            return state, act, r, state1, terminal
        except Exception as e:
            raise e
            print_exc()
            set_trace()

    def getstate_new(self, state):
        # li = [s.state for s in self.sars[-3:]]
        li = []
        state.flags.writeable = False
        li.append(state)
        return self.hashable(np.vstack(li).reshape((1, 1, 4, 4)))

    def subset(self, i0, i1):
        n = i1 - i0
        while 1:
            idxs = [self.idxs[i] for i in range(i0, i1)]
            i0_ = min(idxs) - 3
            i1_ = idxs[-1] + 1
            sars = [self.sars[x] for x in range(i0_, i1_)]
            idxs = [(i-idxs[0]+3) for i in idxs]
            sa = SARli()
            sa.sars = sars
            sa.idxs = idxs
            if len(sa) == n:
                assert idxs[-1] <= len(sars), (idxs[-1], len(sars))
                return sa
            i1 += 1

    def __iter__(self):
        for i in self.idxs:
            yield self.sars[i]
