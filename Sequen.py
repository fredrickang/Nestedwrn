import tensorflow as tf 
import pickle as pk
import argparse
import os
import random
import time
import warnings

from Train import *

parser = argparse.ArgumentParser()

log_dir = 'logs-interf2.pk'
parser.add_argument('--store_dir',default='test',help= 'dir for store result')
parser.add_argument('--data_dir',default= '../Data/cifar-100-python/test-lv3',help = 'dir for test data')
args = parser.parse_args()

fo = open(log_dir,'rb')
logs = pk.load(fo)
fo.close()
acc = []
for i in range(len(logs)):
    for j in range(len(logs[i])):
        testbed = Train()
        acc.append(testbed.test4seq(logs[i][j],args.data_dir))


fo = open('seq_acc-interf2-lv3.pk','wb')
pk.dump(acc, fo)
fo.close()
