import tensorflow as tf 
import pickle as pk
import argparse
import os
import random
import time
import warnings

from Train import *

parser = argparse.ArgumentParser(description="NestedNet training")
parser.add_argument('--evaluate',default= True , help ='evalute model')
parser.add_argument('--store_dir',default='test',help= 'dir for store result')
parser.add_argument('--data_dir',default= '../Data/cifar-100-python/test-lv3',help = 'dir for test data')
args = parser.parse_args()

def testing(testbed, mode, batch_size):
    predic_label = []
    test_image, test_label = read_test_data(args.data_dir)
    num_batch  = 10000/batch_size 
    for step in range(num_batch):
        offset = step*batch_size
        test_batch_image = test_image[offset:offset+batchsize,...]
        if batch_size == 1:
            test_batch_image = test_batch_image.reshape(1,32,32,3)
            dummy = np.zeros((124,32,32,3))
            test_batch_image = np.concatenate((test_batch_image,dummy))
        
        prediction = testbed.test(mode,test_batch_image)
        
        for i in range(batch_size):
            predic_label.append(np.argmax(prediction[0][i]))
    correct = 0
    for i in range(10000):
        if test_label[i] == predic_label[i]:
            correct +=1
    
    return correct/10000

if args.evaluate is False:
    print("activate with training mode")
    train = Train()
    train.train()
else:
    print("activate with test mode")
    print(args.store_dir)
    print(args.data_dir)

    acc = []
  
    # mode 3 batch size 125
    testbed_1 = Train() 
    acc.append(testing(testbed_1,3,125))
    # mode 3 batch_size 1
    testbed_2 = Train()
    acc.append(testing(testbed_2,3,1))
    # mode 1 batch_size 125
    testbed_3 = Train()
    acc.append(testing(testbed_3,1,125))
    # mode 1 batch_size 1
    testbed_4 = Train()
    acc.append(testing(testbed_4,1,1))

    fo = open(os.path.join(args.store_dir,'acc.pk'),'wb')
    pk.dump(acc, fo)
    fo.close()

    time = []

    # mode 3 batch size 125
    test4time_1 = Train() 
    time.append(test4time_1.test4time(3,125,FLAGS.test_ckpt_path))
    # mode 3 batch_size 1
    test4time_2 = Train()
    time.append(test4time_1.test4time(3,1,FLAGS.test_ckpt_path))
    # mode 1 batch_size 125
    test4time_3 = Train()
    time.append(test4time_1.test4time(1,125,FLAGS.test_ckpt_path))
    # mode 1 batch_size 1
    test4time_4 = Train()
    time.append(test4time_1.test4time(1,1,FLAGS.test_ckpt_path))

    fo = open(os.path.join(args.store_dir,'time.pk'),'wb')
    pk.dump(time, fo)
    fo.close()