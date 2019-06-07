import numpy as np
import random
import pickle as pk
from gcd import *
from schedfunc import *
from tasksetgen import *

### PARAMETERS ###
set_case = [2,3,4,5,6]
set_size = 10
u_min = 1 - 0.73
u_max = 1 - 0.58
NN_task = (33,19)    

def main():
    # init taskset 
    taskset = taskset_generation(set_case, set_size, u_min, u_max)
    # insert NN_task & simulate 
    for i in range (len(taskset)):
        for j in range (len(taskset[i])):
            taskset[i][j].insert(0,NN_task)
            spec =taskset[i][j]
            readyQ, execQ, taskset = init_sys(spec)
            LCM = LCMmerge(taskset)
            for time in range(int(LCM)):
                execQ = execute(execQ) #3
                execQ = terminate_job(execQ) #2
                readyQ, execQ = release_job(time, readyQ, execQ) #1 & 1-1
                
