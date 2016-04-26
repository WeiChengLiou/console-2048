#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
from Agent import NNQ, ANN2, CNN
import tensorflow as tf
from utils import encState


class testobj(TestCase):
    def testANN(self):
        obj = NNQ()

    def testtf(self):
        N_BATCH = 1
        ANN2(N_BATCH)
        CNN(N_BATCH)

    def testpredict(self):
        state0 = [
            [0, 0, 2, 4],
            [0, 2, 2, 4],
            [4, 4, 8, 16],
            [16, 32, 64, 128],
            ]
        # state0 = encState(state0)
        obj = NNQ()
        print obj.predict(state0)

        obj.update(state0, 10.)

