#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
import console2048


class testobj(TestCase):
    def testreward(self):
        grid0 = [[8,2,2,0], [4,4,0,0], [0,2,0,0], [0,0,0,0]]
        grid1 = [[8,4,0,0], [8,0,0,0], [2,0,0,0], [0,0,0,2]]
        obj = console2048.Game()
        obj.grid0 = grid0
        r = obj.reward(grid1)
        assert r == 12, r
