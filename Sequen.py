import tensorflow as tf 
import pickle as pk
import argparse
import os
import random
import time
import warnings

from Train import *


log_dir = 'logs are here!'
parser.add_argument('--store_dir',default='test',help= 'dir for store result')
parser.add_argument('--data_dir',default= '../Data/cifar-100-python/test-lv3',help = 'dir for test data')
args = parser.parse_args()

fo = open(log_dir,'rb')
logs = pk.load(fo)
fo.close()

testbed = Train()
acc = testbed.test4seq(logs,args.data_dir)

fo = open('seq_acc.pk','wb')
pk.dump(acc, fo)
fo.close()