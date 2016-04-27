#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
import numpy as np
import itertools as it
from pdb import set_trace


def test(fi, size):
    f = h5py.File(fi, 'w')
    dset = f.create_dataset('DS1', size, maxshape=(None, None, None),
                            chunks=(1, 3, 3), compression='gzip',
                            dtype='f')
    for i in range(10):
        dset[i, :, :] = np.ones((1, 3, 3)) * i

    f.close()


def read(fi):
    with h5py.File(fi, 'r') as f:
        dset = f['DS1']
        print dset[:]


class SaveObj(object):
    def __init__(self, fi, objdic, times):
        self.t = 0
        self.maxt = times
        f = h5py.File(fi, 'w')
        dic = {}
        for p, size in objdic:
            size1 = list(it.chain([times], size))
            # maxshape = [None] * len(size1)
            dset = f.create_dataset(
                p, size1, compression='gzip', dtype='f')
            dic[p] = dset
        self.f = f
        self.dic = dic

    def save(self, pdic):
        for name, mat in pdic:
            dset = self.dic[name]
            dset[self.t, ...] = mat
        self.t = min(self.t + 1, self.maxt)


if __name__ == '__main__':
    test('test.h5', [10, 3, 3])
    read('test.h5')
