import tensorflow as tf 
from Train import *
import pickle as pk

# all task level 3 
#       batch size = 125
testbed_1 = Train()
time_log , err_log =testbed_1.test(2,125,FLAGS.ckpt_path)
logs = [time_log, err_log]
fo = open('3-125.pk','wb')
pk.dump(logs,fo)
fo.close()

#       batch size = 1
testbed_2 = Train()
time_log, err_log = testbed_2.test(2,1,FLAGS.ckpt_path)
logs = [time_log, err_log]
fo = open('3-1.pk','wb')
pk.dump(logs,fo)
fo.close()

# all task level 1
#       batch size = 125
testbed_3 = Train()
time_log, err_log = testbed_3.test(0,125,FLAGS.ckpt_path)
logs = [time_log, err_log]
fo = open('1-125.pk','wb')
pk.dump(logs,fo)
fo.close()

#       batch size = 1
testbed_4 = Train()
time_log, err_log = testbed_4.test(0,1,FLAGS.ckpt_path)
logs = [time_log, err_log]
fo = open('1-1.pk','wb')
pk.dump(logs,fo)
fo.close()