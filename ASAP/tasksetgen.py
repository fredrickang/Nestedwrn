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

def util2pe(util):
    period = random.randrange(10,100,1)
    execution = max(1, period * util)
    return (period, int(execution))

def taskset_generation(set_case, set_size, u_min, u_max):
    res_total = []
    for i in range (len(set_case)):
        res_case = []
        num = set_case[i]
        temp = int(u_min * 100)
        temp2 = int(u_max * 100)
        for j in range (set_size):
            res_taskset = []
            temp3 = random.randrange(temp, temp2)
            temp3 = float(temp3) /100
            temp4 = uunifast(set_case[i], temp3, 1)
            res_util = temp4[0]
            for k in range (set_case[i]):
                (a,b) = util2pe(res_util[k])
                res_taskset.append((a,b))
            res_case.append(res_taskset)
        res_total.append(res_case)
    
    return res_total