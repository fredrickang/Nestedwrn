#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('run', 'hyperparameters.ipynb')


# In[2]:


BN_EPSILON = 0.00001
BN_DECAY = 0.999

## channel scheduling factors (now, 3 levels for conv scheduling)
l2r = 1.0/2.0    # actual density: (l2r^2) *100
l1r = 1.0/4.0    # actual density: (l1r^2) *100


# In[3]:


def create_variables(name, shape):

    n1 = np.sqrt(6. / (shape[0] * shape[1] * l1r * (shape[-2] + shape[-1])))
    n2 = np.sqrt(6. / (shape[0] * shape[1] * l2r * (shape[-2] + shape[-1])))
    n3 = np.sqrt(6. / (shape[0] * shape[1] * (shape[-2] + shape[-1])))

    shape1 = [shape[0], shape[1], int(l1r * shape[2]), int(l1r * shape[3])]
    shape2_1 = [shape[0], shape[1], int((l2r - l1r) * shape[2]), int(l1r * shape[3])]
    shape2_2 = [shape[0], shape[1], int(l2r * shape[2]), int((l2r - l1r) * shape[3])]
    shape3_1 = [shape[0], shape[1], int((1. - l2r) * shape[2]), int(l2r * shape[3])]
    shape3_2 = [shape[0], shape[1], shape[2], int((1. - l2r) * shape[3])]

    lv1_variables = tf.get_variable(name + '_l1', initializer=tf.random_uniform(shape1, -n3, n3, tf.float32, seed=None))
    lv2_1_variables = tf.get_variable(name + '_l2_1', initializer=tf.random_uniform(shape2_1, -n3, n3, tf.float32, seed=None))
    lv2_2_variables = tf.get_variable(name + '_l2_2', initializer=tf.random_uniform(shape2_2, -n3, n3, tf.float32, seed=None))
    lv3_1_variables = tf.get_variable(name + '_l3_1', initializer=tf.random_uniform(shape3_1, -n3, n3, tf.float32, seed=None))
    lv3_2_variables = tf.get_variable(name + '_l3_2', initializer=tf.random_uniform(shape3_2, -n3, n3, tf.float32, seed=None))

    return lv1_variables, lv2_1_variables, lv2_2_variables, lv3_1_variables, lv3_2_variables


# In[4]:


def output_layer(input1, input2, input3, num_labels):

    input_dim1 = input1.get_shape().as_list()[-1]
    input_dim2 = input2.get_shape().as_list()[-1]
    input_dim3 = input3.get_shape().as_list()[-1]

    fc_w1 = tf.get_variable('fc_weights_l1', shape=[input_dim1, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w2 = tf.get_variable('fc_weights_l2', shape=[input_dim2, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))
    fc_w3 = tf.get_variable('fc_weights_l3', shape=[input_dim3, num_labels], initializer=tf.initializers.variance_scaling(scale=1.0))

    fc_b1 = tf.get_variable(name='fc_bias_l1', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b2 = tf.get_variable(name='fc_bias_l2', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_b3 = tf.get_variable(name='fc_bias_l3', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h1 = tf.matmul(input1, fc_w1) + fc_b1
    fc_h2 = tf.matmul(input2, fc_w2) + fc_b2
    fc_h3 = tf.matmul(input3, fc_w3) + fc_b3
    return fc_h1, fc_h2, fc_h3


# In[5]:


def batch_normalization_layer(name, input_layer, dimension, is_training=True):

    beta = tf.get_variable(name + 'beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name + 'gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    mu = tf.get_variable(name + 'mu', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
    sigma = tf.get_variable(name + 'sigma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

    if is_training is True:
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        train_mean = tf.assign(mu, mu * BN_DECAY + mean * (1 - BN_DECAY))
        train_var = tf.assign(sigma, sigma * BN_DECAY + variance * (1 - BN_DECAY))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    else:
        bn_layer = tf.nn.batch_normalization(input_layer, mu, sigma, beta, gamma, BN_EPSILON)

    return bn_layer


# In[6]:


def conv_layer(input_layer, filter_shape, stride, is_training):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer('l1_l2_l3', input_layer, in_channel, is_training)
    bn_layer = tf.nn.relu(bn_layer)

    n1 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l1r * filter_shape[-1])))
    n2 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + l2r * filter_shape[-1])))
    n3 = np.sqrt(6. / (filter_shape[0] * filter_shape[1] * (filter_shape[-2] + filter_shape[-1])))
    filter1 = tf.get_variable('conv_l1',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int(l1r*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))
    filter2 = tf.get_variable('conv_l2',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int((l2r-l1r)*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))
    filter3 = tf.get_variable('conv_l3',
                              initializer=tf.random_uniform([filter_shape[0], filter_shape[1], filter_shape[2], int((1.-l2r)*filter_shape[3])],
                                                            -n3, n3,   tf.float32, seed=None))

    conv1 = tf.nn.conv2d(bn_layer, filter1, strides=[1, stride, stride, 1], padding='SAME')
    conv2 = tf.concat((conv1, tf.nn.conv2d(bn_layer, filter2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((conv2, tf.nn.conv2d(bn_layer, filter3, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3


# In[7]:


def bn_relu_conv_layer(input1, input2, input3, filter_shape, stride, is_training):

    in_channel1 = input1.get_shape().as_list()[-1]
    in_channel2 = input2.get_shape().as_list()[-1]
    in_channel3 = input3.get_shape().as_list()[-1]

    bn_layer1 = batch_normalization_layer('l1', input1, in_channel1, is_training)
    bn_layer1 = tf.nn.relu(bn_layer1)
    bn_layer2 = batch_normalization_layer('l2', input2, in_channel2, is_training)
    bn_layer2 = tf.nn.relu(bn_layer2)
    bn_layer3 = batch_normalization_layer('l3', input3, in_channel3, is_training)
    bn_layer3 = tf.nn.relu(bn_layer3)

    filter1, filter2_1, filter2_2, filter3_1, filter3_2 = create_variables(name='conv', shape=filter_shape)

    conv1 = tf.nn.conv2d(bn_layer1, filter1, strides=[1, stride, stride, 1], padding='SAME')
    conv2 = tf.concat((tf.add(tf.nn.conv2d(bn_layer2[:, :, :, :int(l1r * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer2[:, :, :, int(l1r * filter_shape[2]):int(l2r * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1],
                                           padding='SAME')),
                       tf.nn.conv2d(bn_layer2, filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((tf.add(tf.nn.conv2d(bn_layer3[:, :, :, :int(l1r * filter_shape[2])], filter1, strides=[1, stride, stride, 1], padding='SAME'),
                              tf.nn.conv2d(bn_layer3[:, :, :, int(l1r * filter_shape[2]):int(l2r * filter_shape[2])], filter2_1, strides=[1, stride, stride, 1],
                                           padding='SAME')),
                       tf.nn.conv2d(bn_layer3[:, :, :, :int(l2r * filter_shape[2])], filter2_2, strides=[1, stride, stride, 1], padding='SAME')), 3)
    conv3 = tf.concat((tf.add(conv3, tf.nn.conv2d(bn_layer3[:, :, :, int(l2r * filter_shape[2]):], filter3_1, strides=[1, stride, stride, 1], padding='SAME')),
                       tf.nn.conv2d(bn_layer3, filter3_2, strides=[1, stride, stride, 1], padding='SAME')), 3)

    return conv1, conv2, conv3


# In[8]:


def residual_block(input1, input2, input3, output_channel, wide_scale, is_training, first_block=False):

    input_channel = input3.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    output_channel = int(output_channel * wide_scale)

    if input_channel * wide_scale == output_channel:
        increase_dim = True
        stride = 1
    else:
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv1, conv2, conv3 = conv_layer(input1, [3, 3, input_channel, output_channel], stride, is_training)
        else:
            conv1, conv2, conv3 = bn_relu_conv_layer(input1, input2, input3, [3, 3, input_channel, output_channel], stride, is_training)

    with tf.variable_scope('conv2_in_block'):
        conv1, conv2, conv3 = bn_relu_conv_layer(conv1, conv2, conv3, [3, 3, output_channel, output_channel], 1, is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        if input_channel * wide_scale == output_channel:
            if first_block:
                np0 = int((output_channel * l1r - input_channel) / 2)
                np1 = int((output_channel * l2r - input_channel) / 2)
                np2 = int((output_channel * 1 - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np0, np0]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np2, np2]])
            else:
                np1 = int((output_channel - input_channel) / 2 * l1r)
                np2 = int((output_channel - input_channel) / 2 * l2r)
                np3 = int((output_channel - input_channel) / 2)
                padded_input1 = tf.pad(input1, [[0, 0], [0, 0], [0, 0], [np1, np1]])
                padded_input2 = tf.pad(input2, [[0, 0], [0, 0], [0, 0], [np2, np2]])
                padded_input3 = tf.pad(input3, [[0, 0], [0, 0], [0, 0], [np3, np3]])
        else:
            pooled_input1 = tf.nn.avg_pool(input1, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input1 = tf.pad(pooled_input1, [[0, 0], [0, 0], [0, 0], [int(input_channel*l1r) // 2,
                                                                            int(input_channel*l1r) // 2]])
            pooled_input2 = tf.nn.avg_pool(input2, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input2 = tf.pad(pooled_input2, [[0, 0], [0, 0], [0, 0], [int(input_channel*l2r) // 2,
                                                                            int(input_channel*l2r) // 2]])
            pooled_input3 = tf.nn.avg_pool(input3, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input3 = tf.pad(pooled_input3, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                            input_channel // 2]])
    else:
        padded_input1 = input1
        padded_input2 = input2
        padded_input3 = input3

    output1 = conv1 + padded_input1
    output2 = conv2 + padded_input2
    output3 = conv3 + padded_input3

    return output1, output2, output3


# In[9]:


def inference(input_tensor_batch, n, wide_scale, is_train, reuse):

    layers1 = []
    layers2 = []
    layers3 = []

    with tf.variable_scope('conv0', reuse=reuse):
        std = np.sqrt(6. / (3 * 3 * (3 + 16)))
        filter0 = tf.get_variable('conv_l1_l2_l3', initializer=tf.random_uniform([3, 3, 3, 16], -std, std, tf.float32, seed=None))
        conv0 = tf.nn.conv2d(input_tensor_batch, filter0, strides=[1, 1, 1, 1], padding='SAME')
        layers1.append(conv0)
        layers2.append(conv0)
        layers3.append(conv0)

    for i in range(n):
        with tf.variable_scope('block1_%d' %i, reuse=reuse):
            if i == 0:
                l1conv1, l2conv1, l3conv1 = residual_block(layers1[-1], layers2[-1], layers3[-1], 16, wide_scale, is_train, first_block=True)
            else:
                l1conv1, l2conv1, l3conv1 = residual_block(layers1[-1], layers2[-1], layers3[-1], 16, wide_scale, is_train)
        layers1.append(l1conv1)
        layers2.append(l2conv1)
        layers3.append(l3conv1)

    for i in range(n):
        with tf.variable_scope('block2_%d' %i, reuse=reuse):
            l1conv2, l2conv2, l3conv2 = residual_block(layers1[-1], layers2[-1], layers3[-1], 32, wide_scale, is_train)
        layers1.append(l1conv2)
        layers2.append(l2conv2)
        layers3.append(l3conv2)

    for i in range(n):
        with tf.variable_scope('block3_%d' %i, reuse=reuse):
            l1conv3, l2conv3, l3conv3 = residual_block(layers1[-1], layers2[-1], layers3[-1], 64, wide_scale, is_train)
        layers1.append(l1conv3)
        layers2.append(l2conv3)
        layers3.append(l3conv3)

        assert l3conv3.get_shape().as_list()[1:] == [8, 8, 64*wide_scale]


    with tf.variable_scope('fc', reuse=reuse):
        in_channel1 = layers1[-1].get_shape().as_list()[-1]
        in_channel2 = layers2[-1].get_shape().as_list()[-1]
        in_channel3 = layers3[-1].get_shape().as_list()[-1]

        bn_layer1 = batch_normalization_layer('l1', layers1[-1], in_channel1)
        bn_layer2 = batch_normalization_layer('l2', layers2[-1], in_channel2)
        bn_layer3 = batch_normalization_layer('l3', layers3[-1], in_channel3)

        relu_layer1 = tf.nn.relu(bn_layer1)
        relu_layer2 = tf.nn.relu(bn_layer2)
        relu_layer3 = tf.nn.relu(bn_layer3)

        global_pool1 = tf.reduce_mean(relu_layer1, [1, 2])
        global_pool2 = tf.reduce_mean(relu_layer2, [1, 2])
        global_pool3 = tf.reduce_mean(relu_layer3, [1, 2])

        assert global_pool3.get_shape().as_list()[-1:] == [64*wide_scale]
        output1, output2, output3 = output_layer(global_pool1, global_pool2, global_pool3, 100)


    return output1, output2, output3


# In[ ]:




