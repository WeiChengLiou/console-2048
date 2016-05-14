Console 2048
==============================

A simple implementation of 2048 that runs in a terminal window.
The result of an afternoon messing around.

Defaults to 4x4.  Different grid sizes can be played by passing the width and height as command line arguments.

Ex:
./console2048.py

Parameter:

-n: number of games. (default: 1)
--cols: (default: 4)
--rows: (default: 4)
--agent: manual, random, neural. (default: manual)
--noshow: NOT display game process. (default: 0)
--train: train mode
--ckpt: load check point (not finished)
--config: load config file

Parameter for neural agent:

algo: choose NN structure
alpha: learning rate
gamma: discount rate
epsilon: probability to random step
