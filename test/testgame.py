#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
from console2048 import main
from utils import encState
import copy


class testobj(TestCase):
    def setUp(self):
        self.args = {
            'cols': 4,
            'rows': 4,
            'n': 1,
            'noshow': 0,
            'train': 0,
            }

    def _testinit(self):
        args = copy.deepcopy(self.args)
        args['agent'] = 'random'
        args['noshow'] = 1
        args['n'] = 2
        main(**args)

    def _testencState(self):
        state = [
            [0, 0, 2, 4],
            [0, 2, 2, 4],
            [4, 4, 8, 16],
            [16, 32, 64, 128],
            ]
        encState(state)

    def testNeural(self):
        args = copy.deepcopy(self.args)
        args['agent'] = 'neural'
        args['noshow'] = 1
        args['n'] = 1
        main(**args)

