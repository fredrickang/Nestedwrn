import numpy as np
import random
import pickle as pk

def uunifast(n, u, nsets):
    sets = []
    while len(sets) < nsets:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)

        if all(ut <= 1 for ut in utilizations):
            sets.append(utilizations)

    return sets