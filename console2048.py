from __future__ import print_function

import itertools as it
import os
import sys
import copy
import random
import functools
from Agent import Model, Random, NNQ
from pdb import set_trace
from math import pow
keypad = "adws"
NUMSET = [pow(2, i) for i in range(12)]
NUMSET[0] = 0


#Python 2/3 compatibility.
if sys.version_info[0] == 2:
    range = xrange
    input = raw_input


def _getch_windows(prompt):
    """
    Windows specific version of getch.  Special keys like arrows actually post
    two key events.  If you want to use these keys you can create a dictionary
    and return the result of looking up the appropriate second key within the
    if block.
    """
    print(prompt, end="")
    key = msvcrt.getch()
    if ord(key) == 224:
        key = msvcrt.getch()
        return key
    print(key.decode())
    return key.decode()


def _getch_linux(prompt):
    """Linux specific version of getch."""
    print(prompt, end="")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    char = None
    try:
        char = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
    print(char)
    return char


#Set version of getch to use based on operating system.
if sys.platform[:3] == 'win':
    import msvcrt
    getch = _getch_windows
else:
    import termios
    getch = _getch_linux


class Manual(Model):
    # Manual Player
    def predict(self, state):
        return keypad.index(getch("Enter direction (w/a/s/d): "))

    def update(self, s1, r):
        return self

    def reset(self):
        """"""


def push_row(row, left=True):
    """Push all tiles in one row; like tiles will be merged together."""
    r = 0
    row = row[:] if left else row[::-1]
    new_row = [item for item in row if item]
    for i in range(len(new_row)-1):
        if new_row[i] and new_row[i] == new_row[i+1]:
            r += new_row[i] * 2
            new_row[i], new_row[i+1:] = new_row[i]*2, new_row[i+2:]+[0]
    new_row += [0]*(len(row)-len(new_row))
    return (new_row if left else new_row[::-1]), r


def get_column(grid, column_index):
    """Return the column from the grid at column_index  as a list."""
    return [row[column_index] for row in grid]


def set_column(grid, column_index, new):
    """
    Replace the values in the grid at column_index with the values in new.
    The grid is changed inplace.
    """
    for i,row in enumerate(grid):
        row[column_index] = new[i]


def push_all_rows(grid, left=True):
    """
    Perform a horizontal shift on all rows.
    Pass left=True for left and left=False for right.
    The grid will be changed inplace.
    """
    r = 0
    for i,row in enumerate(grid):
        grid[i], r_ = push_row(row, left)
        r += r_
    return r


def push_all_columns(grid, up=True):
    """
    Perform a vertical shift on all columns.
    Pass up=True for up and up=False for down.
    The grid will be changed inplace.
    """
    r = 0
    for i,val in enumerate(grid[0]):
        column = get_column(grid, i)
        new, r_ = push_row(column, up)
        r += r_
        set_column(grid, i, new)
    return r


def get_empty_cells(grid):
    """Return a list of coordinate pairs corresponding to empty cells."""
    empty = []
    for j,row in enumerate(grid):
        for i,val in enumerate(row):
            if not val:
                empty.append((j,i))
    return empty


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    if get_empty_cells(grid):
        return True
    for row in grid:
        if any(row[i]==row[i+1] for i in range(len(row)-1)):
            return True
    for i,val in enumerate(grid[0]):
        column = get_column(grid, i)
        if any(column[i]==column[i+1] for i in range(len(column)-1)):
            return True
    return False


def get_start_grid(cols=4, rows=4):
    """Create the start grid and seed it with two numbers."""
    grid = [[0]*cols for i in range(rows)]
    for i in range(2):
        empties = get_empty_cells(grid)
        y,x = random.choice(empties)
        grid[y][x] = 2 if random.random() < 0.9 else 4
    return grid


def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = get_empty_cells(grid)
    y,x = random.choice(empties)
    grid[y][x] = 2 if random.random() < 0.9 else 4
    return any_possible_moves(grid)


def print_grid(grid, score):
    """Print a pretty grid to the screen."""
    print("")
    print(score)
    wall = "+------"*len(grid[0])+"+"
    print(wall)
    for row in grid:
        meat = "|".join("{:^6}".format(val) for val in row)
        print("|{}|".format(meat))
        print(wall)


def getNset(grid):
    ret = [0] * len(NUMSET)

    def inc(x):
        i = NUMSET.index(x)
        ret[i] += 1

    map(inc, it.chain(*grid))
    return ret


class Game:
    def __init__(self, cols=4, rows=4, **kwargs):
        self.grid = get_start_grid(cols, rows)
        self.moves = [
            functools.partial(push_all_rows, left=True),
            functools.partial(push_all_rows, left=False),
            functools.partial(push_all_columns, up=True),
            functools.partial(push_all_columns, up=False)]
        self.score = 0
        self.turns = 0
        self.end = False
        self.agent = kwargs['agent']

    def move(self, direction):
        self.turns += 1
        grid_copy = copy.deepcopy(self.grid)
        self.reward = self.moves[direction](self.grid)
        self.grid0 = grid_copy

        self.agent.update(self.grid, self.reward)

        if self.grid == grid_copy:
            return 0
        if prepare_next_turn(self.grid):
            self.score += self.reward
            return self.reward

        self.end = True
        return 0

    def display(self, noshow=True):
        if noshow:
            return
        print_grid(self.grid, self.score)

    def action(self):
        return keypad[self.agent.predict(self.grid)]

    def update(self):
        self.agent.update(self.grid, self.reward)

    def reset(self):
        self.agent.reset()


def initAgent(agent):
    print(agent)
    if agent == 'random':
        return Random()
    elif agent == 'neural':
        return NNQ()
    else:
        return Manual()


def main(**kwargs):
    """
    Get user input.
    Update game state.
    Display updates to user.
    """
    agent = initAgent(kwargs['agent'].lower())
    kwargs['agent'] = agent

    def mainsub(*args):
        game = Game(**kwargs)
        game.display(kwargs['noshow'])
        while True:
            # get_input = getch("Enter direction (w/a/s/d): ")
            get_input = game.action()
            if get_input in keypad:
                game.move(keypad.index(get_input))
                game.update()
            # elif get_input == "q":
            #     break
            # else:
            #     print("\nInvalid choice.")
            #     continue
            if game.end:
                game.display(kwargs['noshow'])
                print("You Lose!", game.turns, game.score)
                break
            game.display(kwargs['noshow'])
        game.reset()

    map(mainsub, range(kwargs['n']))
    print("Thanks for playing.")


def getargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int, help='agent')
    parser.add_argument('--cols', default=4)
    parser.add_argument('--rows', default=4)
    parser.add_argument('--agent', default='manual', help='agent')
    parser.add_argument('--noshow', default=0, help='no display game process')
    return parser


if __name__ == "__main__":
    parser = getargs()
    args = parser.parse_args()
    args = vars(args)
    main(**args)
