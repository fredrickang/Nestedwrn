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
parser.add_argument('--store_dir',default='test',help= 'dir for store result')
parser.add_argument('--data_dir',default= '../Data/cifar-100-python/test-lv3',help = 'dir for test data')
args = parser.parse_args()


if args.evaluate is False:
    print("activate with training mode")
    train = Train()
    train.train()
else:
    print("activate with test mode")
    print(args.store_dir)
    print(args.data_dir)
    '''
    acc = []
  
    # mode 3 batch size 125
    #testbed_1 = Train() 
    #acc.append(testbed_1.test(125,3,args.data_dir))
    # mode 3 batch_size 1
    testbed_2 = Train()
    acc.append(testbed_2.test(1,3,args.data_dir))
    # mode 1 batch_size 125
    #testbed_3 = Train()
    #acc.append(testbed_3.test(125,1,args.data_dir))
    # mode 1 batch_size 1
    testbed_4 = Train()
    acc.append(testbed_4.test(1,1,args.data_dir))

    fo = open(os.path.join(args.store_dir,'acc2.pk'),'wb')
    pk.dump(acc, fo)
    fo.close()
    '''
    time = []

    # mode 3 batch size 125
    test4time_1 = Train() 
    time.append(test4time_1.test4time(3,125,FLAGS.test_ckpt_path,args.data_dir))
    # mode 3 batch_size 1
    test4time_2 = Train()
    time.append(test4time_2.test4time(3,1,FLAGS.test_ckpt_path,args.data_dir))
    # mode 2 batch_size 125
    test4time_5 = Train()
    time.append(test4time_5.test4time(2,125,FLAGS.test_ckpt_path,args.data_dir))
    # mode 2 batch_size 1
    test4time_6 = Train()
    time.append(test4time_6.test4time(2,1,FLAGS.test_ckpt_path,args.data_dir))
    # mode 1 batch_size 125
    test4time_3 = Train()
    time.append(test4time_3.test4time(1,125,FLAGS.test_ckpt_path,args.data_dir))
    # mode 1 batch_size 1
    test4time_4 = Train()
    time.append(test4time_4.test4time(1,1,FLAGS.test_ckpt_path,args.data_dir))
  
    

    fo = open(os.path.join(args.store_dir,'time.pk'),'wb')
    pk.dump(time, fo)
    fo.close()
    