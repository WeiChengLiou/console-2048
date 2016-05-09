#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
from Agent import NNQ, ANN, CNN
import tensorflow as tf
from utils import encState


class testobj(TestCase):
    def testANN(self):
        obj = NNQ()

    def testtf(self):
        N_BATCH = 1
        ANN(N_BATCH)
        CNN(N_BATCH)

    def chkeq(self, mat1, mat2):
        cnt, N = 0, mat1.shape[0]
        for i in range(N):
            cnt += int(all([(x1 == x2) for x1, x2 in
                       zip(mat1[i, :], mat2[i, :])]))
        return cnt == N

    def testpredict(self):
        state0 = [
            [0, 0, 2, 4],
            [0, 2, 2, 4],
            [4, 4, 8, 16],
            [16, 32, 64, 128],
            ]
        # state0 = encState(state0)
        obj = NNQ()

        mat, = obj.sess.run([obj.fc1_weights])
        obj.save()
        print obj.predict(state0)

        obj.update(state0, 10.)
        mat1, = obj.sess.run([obj.fc1_weights])
        self.assertFalse(self.chkeq(mat, mat1))

        obj.load()
        mat1, = obj.sess.run([obj.fc1_weights])
        self.assertTrue(self.chkeq(mat, mat1))
