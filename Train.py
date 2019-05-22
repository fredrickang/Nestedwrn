#!/usr/bin/env python
# coding: utf-8

# In[2]:



from WRN import *
from input_module import *
import time
import tensorflow as tf
import pandas as pd
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


    with tf.device('/GPU:0'): 
        def train(self):
            best_acc1 = 0
            all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
            vali_data, vali_labels = read_validation_data()
                
            self.build_train_validation_graph()

                
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
            summary_op = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True 
            sess = tf.Session(config=config)

            if FLAGS.is_use_ckpt is True:
                saver.restore(sess, FLAGS.ckpt_path)
                print ('Restored from checkpoint...')
            else:
                sess.run(init)

            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

            step_list = []
            train_error_list_1 = []
            train_error_list_2 = []
            train_error_list_3 = []
            val_error_list_1 = []
            val_error_list_2 = []
            val_error_list_3 = []




            print ('Start training...')
            print ('-------------------------------------------------------------------------------------------')
            start_time = time.time()
            for epoch in range(1,FLAGS.train_epochs+1):
                for step in range(int(EPOCH_SIZE/FLAGS.train_batch_size)):
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
                    """    
                    if step % FLAGS.report_freq == 0 and step > 0:
                        print(
                            "epoch %3d, step %3d :  Train loss1 = %.3f, Train acc1 = %.3f\n"
                            "                       Train loss2 = %.3f, Train acc2 = %.3f\n"
                            "                       Train loss3 = %.3f, Train acc3 = %.3f, cumulative time = %.3f sec\n"
                            "-------------------------------------------------------------------------------------------------------------------------------------------"
                            % (epoch, step,
                                            tr_l1, 1 - tr_e1,
                                            tr_l2, 1 - tr_e2, 
                                            tr_l3, 1 - tr_e3, time.time() - start_time))
                    """
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

                #vali_summ1 = tf.Summary()
                #vali_summ1.value.add(tag = 'full_validation_err1',simple_value =val_e1.astype(np.float))
                #summary_writer.add_summary(vali_summ1, epoch*FLAGS.train_batch_size+step)
                #summary_writer.flush()
                    
                #vali_summ2 = tf.Summary()
                #vali_summ2.value.add(tag = 'full_validation_err2',simple_value =val_e2.astype(np.float))
                #summary_writer.add_summary(vali_summ2,epoch*FLAGS.train_batch_size + step)
                #summary_writer.flush()

                #vali_summ3 = tf.Summary()
                #vali_summ3.value.add(tag = 'full_validation_err3',simple_value = val_e3.astype(np.float))
                #summary_writer.add_summary(vali_summ3,epoch*FLAGS.train_batch_size + step)
                #summary_writer.flush()

                # summary_str = sess.run([summary_op],{self.image_placeholder: train_batch_data,
                #                         self.label_placeholder: train_batch_labels,
                #                         self.vali_image_placeholder: vali_batch_data,
                #                         self.vali_label_placeholder: vali_batch_labels,
                #                         self.lr_placeholder: FLAGS.init_lr})
                # summary_writer.add_summary(summary_str, epoch*FLAGS.train_batch_size + step)
                
                print(
                    "epoch %3d: Val loss1 = %.3f,Val acc1 = %.3f (WRN-%d-%d), time = %.3f  \n"
                    "           Val loss2 = %.3f,Val acc2 = %.3f (WRN-%d-%d), time = %.3f  \n"
                    "           Val loss3 = %.3f,Val acc3 = %.3f (WRN-%d-%d), time = %.3f, cumulative time = %.3f sec\n"
                    "-------------------------------------------------------------------------------------------------------------------------------------------"
                    % (epoch,
                       val_l1, 1 - val_e1, FLAGS.res_blocks*6+2, 1,time1,
                    val_l2, 1 - val_e2, FLAGS.res_blocks*6+2, FLAGS.wide_factor/2, time2,
                    val_l3, 1 - val_e3, FLAGS.res_blocks*6+2, FLAGS.wide_factor, time3, time.time() - start_time))
                    
                       
                step_list.append((epoch-1)*FLAGS.train_batch_size+step)
                    
                train_error_list_1.append(tr_e1)
                train_error_list_2.append(tr_e2)
                train_error_list_3.append(tr_e3)

                val_error_list_1.append(val_e1)
                val_error_list_2.append(val_e2)
                val_error_list_3.append(val_e3)

                
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)

                df = pd.DataFrame(data={'step':step_list,'train_err1':train_error_list_1,'train_err2':train_error_list_2,'train_err3':train_error_list_3,
                                            'val_err1':val_error_list_1,'val_err2':val_error_list_2,'val_err3':val_error_list_3})
                df.to_csv(train_dir +FLAGS.version+ '_error.csv')

                if epoch == FLAGS.decay_epoch0 or epoch == FLAGS.decay_epoch1 or epoch ==  FLAGS.decay_epoch2:
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
    '''
    def test(self, mode, test_batch_size ,ckpt_path, test_data_dir):
        mode = mode -1

        self.test_image_placeholder = tf.placeholder(dtype =tf.float32, shape = [test_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[test_batch_size])

        
        logits1, logits2, logits3 = inference(self.test_image_placeholder, FLAGS.res_blocks, FLAGS.wide_factor, True, reuse=False)
        
        predictions = [tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3)]
        self.test_top1_error = self.top_k_error(predictions[mode], self.test_label_placeholder, 1)

        saver = tf.train.Saver(tf.all_variables())
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

        tf.reset_default_graph()
        saver.restore(sess, ckpt_path)
        print('Model restored from ',ckpt_path)
        
        err_list = []
        time_log = []
        test_data, test_labels = read_test_data(test_data_dir)
        for step in range(int(10000/test_batch_size)):
            
            if step % 10 == 0:
                print ('%i batches finished!' %step)
            offset = step * test_batch_size
            test_image_batch = test_data[offset:offset+test_batch_size, ...]
            test_label_batch = test_labels[offset:offset+test_batch_size, ...]
            t = time.time()
            top1_err_val = sess.run([self.test_top1_error],feed_dict={self.test_image_placeholder:test_image_batch, self.test_label_placeholder: test_label_batch})
            t_val = time.time() -t
            time_log.append(t_val)
            err_list.append(top1_err_val)
        sess.close()
        return time_log, err_list

    '''
    def test(self, test_image_array, mode):
        mode = mode -1


        num_test_images = len(test_image_array)
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[num_test_images,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH])

        logits1, logits2, logits3 = inference(self.test_image_placeholder, FLAGS.res_blocks, FLAGS.wide_factor, True, reuse=False)
        predictions = [tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3)]

        saver = tf.train.Saver(tf.all_variables())
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.test_ckpt_path)

        print("Model restored from", FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1,100)

        batch_prediction_array = sess.run([predictions[mode]],feed_dict={self.test_image_placeholder: test_image_array})

        prediction_array = np.concatenate((prediction_array,batch_prediction_array))

        return prediction_array