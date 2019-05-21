import tensorflow as tf 
import pickle as pk
import argparse
import os
import random
import time
import warnings

from Train import *

parser = argparse.ArgumentParser(description="NestedNet training")
parser.add_argument('--evaluate',default= False , help ='evalute model')
parser.add_argument('--store_dir',default= '',help= 'dir for store result')
parser.add_argument('--data_dir',default= '../Data/cifar-100-python/noisy-test',help = 'dir for test data')
args = parser.parse_args()

if args.evaluate is False:
    print("activate with training mode")
    train = Train()
    train.train()
else:
    print("activate with test mode")
    # all task level 3 
    #       batch size = 125
    
    testbed_1 = Train()
    time_log , err_log =testbed_1.test(3,125,FLAGS.ckpt_path,args.data_dir)
    logs = [time_log, err_log]
    fo = open(os.path.join(args.store_dir,'3-125.pk'),'wb')
    pk.dump(logs,fo)
    fo.close()
    #       batch size = 1
    testbed_2 = Train()
    time_log, err_log = testbed_2.test(3,1,FLAGS.ckpt_path,args.data_dir)
    logs = [time_log, err_log]
    fo = open(os.path.join(args.store_dir,'3-1.pk'),'wb')
    pk.dump(logs,fo)
    fo.close()

    # all task level 1
    #       batch size = 125
    testbed_3 = Train()
    time_log, err_log = testbed_3.test(1,125,FLAGS.ckpt_path,args.data_dir)
    logs = [time_log, err_log]
    fo = open(os.path.join(args.store_dir,'1-125.pk'),'wb')
    pk.dump(logs,fo)
    fo.close()

    #       batch size = 1
    testbed_4 = Train()
    time_log, err_log = testbed_4.test(1,1,FLAGS.ckpt_path,args.data_dir)
    logs = [time_log, err_log]
    fo = open(os.path.join(args.store_dir,'1-1.pk'),'wb')
    pk.dump(logs,fo)
    fo.close()