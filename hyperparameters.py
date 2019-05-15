#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


FLAGS = tf.app.flags.FLAGS


# In[3]:


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'test_110', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 10, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.9, '''The decay factor of the train error's
moving average shown on tensorboard''')


# In[4]:


## The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_epochs', 120, '''Total steps that you want to train''')
tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
tf.app.flags.DEFINE_integer('validation_batch_size', 128, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 128, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each time''')
tf.app.flags.DEFINE_integer('decay_epoch0', 30, '''At which epoch to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_epoch1', 60, '''At which epoch to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_epoch2', 90, "At which epoch to decay the learning rate")


# In[5]:


## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('res_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

tf.app.flags.DEFINE_float('wide_factor', 4, '''scale # conv channels in WRN''')


# In[6]:


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 4, '''In data augmentation, layers of zero padding on
each side of the image''')


# In[1]:


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'logs_test_110/model.ckpt-389', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'


# In[ ]:




