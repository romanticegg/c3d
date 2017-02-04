# # convert the original c3d model from the author to a tf 0.12 compatibal model
# fixme: all problematic, subject to removel
# import tensorflow as tf
# import ref_version.c3d_model as c3d_model
# import c3d_input_ucf101 as input_reader
# import tf_utils
# import sys
#
# flags = tf.app.flags
# flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/UCF-101-g16/train', 'directory to save training data[/Users/zijwei/Dev/datasets]')
# flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
# flags.DEFINE_integer('batch_size', 12, 'batch size[12]')
# flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
# flags.DEFINE_integer('max_steps', 100000, 'Number of training steps[100000]')
# flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
# flags.DEFINE_float('init_lr', 0.05, 'initial learning rate[0.05]')
# flags.DEFINE_float('lr_decay_rate', 0.5, 'Decay learning rate by [0.5]')
# flags.DEFINE_integer('num_epoch_per_decay', 4, 'decay of learning rate every [4] epoches')
# flags.DEFINE_float('weight_decay_conv', 0.0005, 'weight decay for convolutional layers [0.0005]')
# flags.DEFINE_float('weight_decay_fc', 0.0005, 'weight decay for fully connected (fully convoluted) layers [0.0005]')
# flags.DEFINE_float('dropout', 0.5, 'dropuout ratio[0.5]')
#
# FLAGS = flags.FLAGS
#
#
#
#
#
# def main(argv=None):
#     tf_utils.print_gflags(FLAGS)
#
#     weights = {
#         'wc1': tf_utils.variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], tf.contrib.layers.xavier_initializer(),0.0005),
#         'wc2': tf_utils.variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], tf.contrib.layers.xavier_initializer(),0.0005),
#         'wc3a': tf_utils.variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256],tf.contrib.layers.xavier_initializer(),0.0005),
#         'wc3b': tf_utils.variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], tf.contrib.layers.xavier_initializer(),0.0005),
#         'wc4a': tf_utils.variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], tf.contrib.layers.xavier_initializer(),0.0005),
#         'wc4b': tf_utils.variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
#         'wc5a': tf_utils.variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
#         'wc5b': tf_utils.variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
#         'wd1': tf_utils.variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
#         'wd2': tf_utils.variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
#         'out': tf_utils.variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
#     }
#     biases = {
#         'bc1': tf_utils.variable_with_weight_decay('bc1', [64], 0.000),
#         'bc2': tf_utils.variable_with_weight_decay('bc2', [128], 0.000),
#         'bc3a': tf_utils.variable_with_weight_decay('bc3a', [256], 0.000),
#         'bc3b': tf_utils.variable_with_weight_decay('bc3b', [256], 0.000),
#         'bc4a': tf_utils.variable_with_weight_decay('bc4a', [512], 0.000),
#         'bc4b': tf_utils.variable_with_weight_decay('bc4b', [512], 0.000),
#         'bc5a': tf_utils.variable_with_weight_decay('bc5a', [512], 0.000),
#         'bc5b': tf_utils.variable_with_weight_decay('bc5b', [512], 0.000),
#         'bd1': tf_utils.variable_with_weight_decay('bd1', [4096], 0.000),
#         'bd2': tf_utils.variable_with_weight_decay('bd2', [4096], 0.000),
#         'out': tf_utils.variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
#     }
#
#     with tf.Graph().as_default() as graph:
#         global_step = tf.get_variable(name='gstep', initializer=tf.constant(0), trainable=False)
#         batch_images, batch_labels, _ = input_reader.inputs(FLAGS.data_dir, isTraining=True)
#         print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
#         print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
#         print '-'*32
#         sys.stdout.flush()
#
#
#         logits= c3d_model.inference_c3d(batch_images, FLAGS.dropout, FLAGS.batch_size, _weights, _biases):
#
#         logits = c3d_model.inference_c3d(batch_images, isTraining=True)
#         loss = c3d_model.loss(logits=logits, labels=batch_labels)
#
#         train_op, lr = c3d_model.train(loss, global_step, lr_decay_every_n_step)
#
#         correct_ones = c3d_model.correct_ones(logits=logits, labels=batch_labels)
#
#         saver = tf.train.Saver(max_to_keep=None)
#         summary_op = tf.summary.merge_all()
#
#         config = tf_utils.gpu_config(FLAGS.gpu_id)
#         with tf.Session(config=config) as sess:
#             sess.run(tf.variables_initializer(tf.global_variables()))
#             summary_writer = tf.summary.FileWriter(logdir=save_locations.summary_save_dir, graph=sess.graph)
#
#             coord = tf.train.Coordinator()
#
#             threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#             print 'Training Start!'
#             sys.stdout.flush()
#
#             cum_loss = 0
#             cum_correct = 0
#
#             for i in range(FLAGS.max_steps):
#                 _, loss_, correct_ones_ = sess.run([train_op, loss, correct_ones])
#
#                 assert not np.isnan(loss_), 'Model diverged with loss = NaN, try again'
#
#                 cum_loss += loss_
#                 cum_correct += correct_ones_
#                 # update: print loss every epoch
#                 if (i+1) % steps_per_epoch == 0:
#                     lr_ = sess.run(lr)
#
#                     print '[{:s} -- {:08d}|{:08d}]\tloss : {:.3f}\t, correct ones [{:d}|{:d}], l-rate:{:.06f}'.format(save_dir, i, FLAGS.max_steps,
#                                                                                           cum_loss/steps_per_epoch, cum_correct, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, lr_)
#                     sys.stdout.flush()
#                     cum_loss = 0
#                     cum_correct = 0
#
#                 if (i+1 % 100) == 0:
#                     summary_ = sess.run(summary_op)
#                     summary_writer.add_summary(summary_, global_step=global_step)
#
#                 if (i+1) % 2000 == 0:
#                     save_path = os.path.join(save_locations.model_save_dir, 'model')
#                     saver.save(sess=sess,global_step=global_step, save_path=save_path)
#             coord.request_stop()
#             coord.join(threads=threads)
#
#
# if __name__ == '__main__':
#     tf.app.run()
#
#
