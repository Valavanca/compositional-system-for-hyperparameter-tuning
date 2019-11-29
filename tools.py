from random import randrange, choice
from collections import namedtuple

import numpy as np
import pandas as pd


def samples(problem, n=10, random=True):
    lin = []
    for b in zip(*problem.get_bounds()):
        lin.append(np.linspace(b[0], b[1], 20+randrange(30), endpoint=True))

    Point = namedtuple('Point', ['x{}'.format(i+1) for i in range(len(lin))])
    table = pd.DataFrame([Point(*[choice(d) for d in lin]) for _ in range(n)])
    table[['f1', 'f2']] = table.apply(
        lambda row: pd.Series(problem.fitness(row)), axis=1)

    if random is True:
        table['rand1'] = np.random.randint(1, 50, table.shape[0])
        table['rand2'] = np.random.randint(1, 100, table.shape[0])
        table['rand3'] = np.random.randint(1, 250, table.shape[0])
        table['rand4'] = np.random.randint(1, 1000, table.shape[0])

    return table


