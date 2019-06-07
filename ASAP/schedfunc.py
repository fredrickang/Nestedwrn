import numpy as np
import random
import pickle as pk

### PARAMETERS ###
UNIT_TIME = 1
MODE1 = 8
MODE2 = 11
MODE3 = 13
### !PARAMETERS ###  

class task():
    def __init__(self, period, execution, name, is_target=False):
        self.p = float(period)
        self.e = float(execution)
        self.t = is_target
        self.name = name
        self.u = self.e/self.p
        
        
class job():
    def __init__(self,release,task):
        self.r = release
        self.task = task
        self.d = self.r + self.task.p
        #self.ae = round((random.randrange(5,10,1)/10.0)* self.task.e,1)
        self.ae = float(self.task.e)
        self.remain = self.ae
        self.mode = 1
    def execute(self):
        self.remain = self.remain - UNIT_TIME
    def __lt__(self, other):
         return self.d < other.d
    def switch_mode(self, mode):
        if mode == 2:
            self.ae = MODE2
            self.remain = MODE2
            self.mode = 2
        if mode ==3:
            self.ae = MODE3
            self.remain = MODE3           
            self.mode = 3 
            
    

def init_sys(spec,spec_size, MODE1=8):
    print("========Init System========")
    readyQ = []
    execQ = []
    
    taskset = create_taskset(spec)
    
    taskset.append(task(33,MODE1,str(spec_size),True))
    
    for tasks in taskset:
        enqueue_job(readyQ, 0,tasks)
    return readyQ, execQ, taskset

def create_taskset(spec):
    taskset = []
    num_task = len(spec)
    for i in range(num_task):
        taskset.append(task(spec[i][0],spec[i][1],str(i)))
    return taskset

def enqueue_job(Q, future_release, task):
    job_instance = job(future_release, task)
    Q.append(job_instance)
    return Q

def release_job(t_cur, readyQ, execQ):
    new_readyQ = []
    for jobs in readyQ:
        if jobs.r == t_cur:
            execQ.append(jobs)
            new_readyQ = enqueue_job(new_readyQ, t_cur+jobs.task.p, jobs.task)
        else:
            new_readyQ.append(jobs)
    return new_readyQ, execQ

def ASAP_release_job(t_cur, readyQ, execQ):
    new_readyQ = []
    NN_released = False
    for jobs in readyQ:
        if jobs.r == t_cur:
            if (jobs.task.t == True):
                NN_released = jobs.task.t
            execQ.append(jobs)
            new_readyQ = enqueue_job(new_readyQ, t_cur+jobs.task.p, jobs.task)
        else:
            new_readyQ.append(jobs)
    return new_readyQ, execQ, NN_released

def terminate_job(execQ):
    new_execQ = []
    for jobs in execQ:
        if jobs.remain > 0:
            new_execQ.append(jobs)
    return new_execQ 

def execute(execQ):
    res_idle = 0
    execQ.sort()
    if len(execQ) == 0:
        res_idle = 1
    else:
        execQ[0].execute()
    return execQ, res_idle

def slack_calc(t_cur, taskset, readyQ, execQ):
    u_tot = 0
    p = 0 
    
    for tasks in taskset:
        u_tot += tasks.u
    
    readyQ_list =[]
    execQ_list = []
    
    sorted_tasks = []
    
    for jobs in readyQ:
        readyQ_list.append(jobs.task.name)
    for jobs in execQ:
        execQ_list.append(jobs.task.name)
    
    for names in execQ_list:
        readyQ_list.remove(names)
    
    for names in readyQ_list:
        for jobs in readyQ:
            if jobs.task.name == names:
                sorted_tasks.append(jobs)
    
    sorted_tasks = sorted_tasks + execQ
    sorted_tasks.sort(key =lambda jobs: jobs.d, reverse= True)
    
    for jobs in sorted_tasks:
        u_tot -= jobs.task.u
        d_diff = jobs.d - sorted_tasks[-1].d
        
        
        if d_diff == 0:
            qi =jobs.remain
            u_tot = 1
        else:
            qi = max(0,(jobs.remain - (1.0-u_tot)*d_diff))
            u_tot = min(1.0,u_tot +(jobs.remain))
        p += qi
    
    slack = sorted_tasks[-1].d - t_cur - p
    
    return slack