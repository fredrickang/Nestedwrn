#!/usr/bin/env python
# coding: utf-8

# In[2]:



from WRN import *
from input_module import *
import time
import tensorflow as tf

# In[1]:


class Train(object):
    def __init__(self):
        self.placeholders()

    def placeholders(self):
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])


    def build_train_validation_graph(self):
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        logits1, logits2, logits3 = inference(self.image_placeholder, FLAGS.res_blocks, FLAGS.wide_factor, True, reuse=False)
        vali_logits1, vali_logits2, vali_logits3 = inference(self.vali_image_placeholder, FLAGS.res_blocks, FLAGS.wide_factor, False, reuse=True)

        t_vars = tf.trainable_variables()

        regu_loss = sum([tf.nn.l2_loss(w) for w in t_vars])

        loss1 = self.loss(logits1, self.label_placeholder)
        loss2 = self.loss(logits2, self.label_placeholder)
        loss3 = self.loss(logits3, self.label_placeholder)

        self.full_loss1 = tf.add_n([loss1]) + FLAGS.weight_decay * regu_loss
        self.full_loss2 = tf.add_n([loss2]) + FLAGS.weight_decay * regu_loss
        self.full_loss3 = tf.add_n([loss3]) + FLAGS.weight_decay * regu_loss

        self.total_loss = 0.5*tf.add_n([loss1]) + 0.3*tf.add_n([loss2]) + 0.2*tf.add_n([loss3]) + FLAGS.weight_decay * regu_loss

        predictions1 = tf.nn.softmax(logits1)
        predictions2 = tf.nn.softmax(logits2)
        predictions3 = tf.nn.softmax(logits3)

        self.train_top1_error1 = self.top_k_error(predictions1, self.label_placeholder, 1)
        self.train_top1_error2 = self.top_k_error(predictions2, self.label_placeholder, 1)
        self.train_top1_error3 = self.top_k_error(predictions3, self.label_placeholder, 1)


        # Validation loss
        self.vali_loss1 = self.loss(vali_logits1, self.vali_label_placeholder)
        vali_predictions1 = tf.nn.softmax(vali_logits1)
        self.vali_top1_error1 = self.top_k_error(vali_predictions1, self.vali_label_placeholder, 1)

        self.vali_loss2 = self.loss(vali_logits2, self.vali_label_placeholder)
        vali_predictions2 = tf.nn.softmax(vali_logits2)
        self.vali_top1_error2 = self.top_k_error(vali_predictions2, self.vali_label_placeholder, 1)

        self.vali_loss3 = self.loss(vali_logits3, self.vali_label_placeholder)
        vali_predictions3 = tf.nn.softmax(vali_logits3)
        self.vali_top1_error3 = self.top_k_error(vali_predictions3, self.vali_label_placeholder, 1)

        self.train_op = self.train_operation(global_step, self.total_loss, self.train_top1_error3, t_vars)


    with tf.device('/cpu:0'):
        def train(self):

            all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
            vali_data, vali_labels = read_validation_data()

            self.build_train_validation_graph()

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            print ('Start training...')
            print ('-------------------------------------------------------------------------------------------')

            for step in range(FLAGS.train_steps):

                train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels, FLAGS.train_batch_size)
                vali_batch_data, vali_batch_labels = self.generate_vali_batch(vali_data, vali_labels, FLAGS.validation_batch_size)

                _ = sess.run([self.train_op],
                             {self.image_placeholder: train_batch_data,
                              self.label_placeholder: train_batch_labels,
                              self.vali_image_placeholder: vali_batch_data,
                              self.vali_label_placeholder: vali_batch_labels,
                              self.lr_placeholder: FLAGS.init_lr})

                tr_l1, tr_e1, tr_l2, tr_e2, tr_l3, tr_e3 = sess.run([self.full_loss1, self.train_top1_error1,
                                                                     self.full_loss2, self.train_top1_error2,
                                                                     self.full_loss3, self.train_top1_error3],
                                                                    {self.image_placeholder: train_batch_data,
                                                                     self.label_placeholder: train_batch_labels,
                                                                     self.vali_image_placeholder: vali_batch_data,
                                                                     self.vali_label_placeholder: vali_batch_labels,
                                                                     self.lr_placeholder: FLAGS.init_lr})

                if step % FLAGS.report_freq == 0 and step > 0:
                    val_l1, val_e1, time1 = self.full_validation(loss=self.vali_loss1, top1_error=self.vali_top1_error1,
                                                                 vali_data=vali_data, vali_labels=vali_labels,
                                                                 session=sess, batch_data=train_batch_data,
                                                                 batch_label=train_batch_labels)

                    val_l2, val_e2, time2 = self.full_validation(loss=self.vali_loss2, top1_error=self.vali_top1_error2,
                                                                 vali_data=vali_data, vali_labels=vali_labels,
                                                                 session=sess, batch_data=train_batch_data,
                                                                 batch_label=train_batch_labels)

                    val_l3, val_e3, time3 = self.full_validation(loss=self.vali_loss3, top1_error=self.vali_top1_error3,
                                                                 vali_data=vali_data, vali_labels=vali_labels,
                                                                 session=sess, batch_data=train_batch_data,
                                                                 batch_label=train_batch_labels)

                    print(
                        "epoch %3d: Train loss1 = %.3f, Val loss1 = %.3f, Train acc1 = %.3f, Val acc1 = %.3f (WRN-%d-%d), time = %.3f  \n"
                        "           Train loss2 = %.3f, Val loss2 = %.3f, Train acc2 = %.3f, Val acc2 = %.3f (WRN-%d-%d), time = %.3f  \n"
                        "           Train loss3 = %.3f, Val loss3 = %.3f, Train acc3 = %.3f, Val acc3 = %.3f (WRN-%d-%d), time = %.3f, cumulative time = %.3f sec\n"
                        "-------------------------------------------------------------------------------------------------------------------------------------------"
                        % (step / FLAGS.report_freq,
                           tr_l1, val_l1, 1 - tr_e1, 1 - val_e1, FLAGS.res_blocks*6+2, 1,time1,
                           tr_l2, val_l2, 1 - tr_e2, 1 - val_e2, FLAGS.res_blocks*6+2, FLAGS.wide_factor/2, time2,
                           tr_l3, val_l3, 1 - tr_e3, 1 - val_e3, FLAGS.res_blocks*6+2, FLAGS.wide_factor, time3, time.time() - start_time))


                if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                    FLAGS.init_lr = 0.1 * FLAGS.init_lr
                    print ('Learning rate decayed to ', FLAGS.init_lr)

            # sys.stdout.close()


    ## Helper functions
    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def compute_sr(self, _theta, th):
        ## compute sparse ratio
        nz_size = all_size = 0
        for i in range(len(_theta)):
            nz_size += np.sum([np.abs(_theta[i]) > th])
            all_size += np.size(_theta[i])
        return float(nz_size) / float(all_size)

    def top_k_error(self, predictions, labels, k):
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label


    def train_operation(self, global_step, total_loss, top1_error, var_lists):
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9, use_nesterov=True)
        train_op = opt.minimize(total_loss, global_step=global_step, var_list=var_lists)
        return train_op


    def validation_op(self, validation_step, top1_error, loss):
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)

        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data, batch_label):
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        t = time.time()
        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)
        t_val = time.time() - t

        return np.mean(loss_list), np.mean(error_list), t_val/num_batches


train = Train()
train.train()
