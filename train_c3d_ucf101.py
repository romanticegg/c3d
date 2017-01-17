# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy as np
import random
import tensorflow as tf

import input_data
import c3d_model
import consts as c


# Basic model parameters as external flags.
flags = tf.app.flags
#todo: refine this section
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer, if 0, test on both training and testing data [5000]')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('gpu_id', None, 'id of GPU')
flags.DEFINE_integer('randomseed', 0, 'random seed to produce reproducable results')
flags.DEFINE_boolean('rgb', True, 'If data should be loaded as RGB[True] or BGR')
# flags.DEFINE_boolean('testonly', False, 'If only testing without training[False]')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels)
    )
    tf.summary.scalar(
        name_scope + 'cross entropy',
        cross_entropy_mean
    )
    weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
    tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses', name_scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar(name_scope + 'total loss', total_loss)

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def performance_eval(sess, tf_acc, images_placeholder, labels_placeholder, file_list, label_list, batch_size, np_mean):
    cum_acc = 0
    cum_files = 0
    n_files = len(file_list)
    n_batches = int(n_files / batch_size)
    for batch_id in xrange(n_batches + 1):
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, n_files)

        batch_files = file_list[start_idx:end_idx]
        batch_labels = label_list[start_idx:end_idx]
        images_data, labels_data = input_data.read_clip_and_label(
            dirnames=batch_files,
            labels=batch_labels,
            np_mean=np_mean,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            RGB= FLAGS.rgb
        )
        if labels_data.size:  # check on nparray
            batch_acc = sess.run(
                tf_acc,
                feed_dict={
                    images_placeholder: images_data,
                    labels_placeholder: labels_data
                })
            cum_files += labels_data.size
            cum_acc += batch_acc * labels_data.size
            # test_writer.add_summary(summary, step)
    return cum_acc * 1.0 / cum_files


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    use_pretrained_model = True
    model_filename = "./models/sports1m_finetuning_ucf101.model"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size
        )
        tower_grads1 = []
        tower_grads2 = []
        # logits = []
        opt1 = tf.train.AdamOptimizer(1e-4)
        opt2 = tf.train.AdamOptimizer(2e-4)
        # for gpu_index in range(0, gpu_num):
        # if FLAGS.gpu_id is not None:
        #   with tf.device('/gpu:%d' % FLAGS.gpu_id):

        # with tf.name_scope('{:s}_{:01d}'.format('zijwei-research', FLAGS.gpu_id)) as scope:
        with tf.name_scope('zijwei-research') as scope:
            with tf.variable_scope('var_name') as var_scope:
                weights = {
                    'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                    'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                    'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                    'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                    'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                    'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                    'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                    'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                    'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                    'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                    'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
                }
                biases = {
                    'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                    'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                    'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                    'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                    'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                    'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                    'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                    'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                    'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                    'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                    'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
                }
                varlist1 = weights.values()
                varlist2 = biases.values()
                logit = c3d_model.inference_c3d(
                    images_placeholder,
                    0.5,
                    FLAGS.batch_size,
                    weights,
                    biases
                )
                loss = tower_loss(
                    scope,
                    logit,
                    labels_placeholder
                )
                grads1 = opt1.compute_gradients(loss, varlist1)
                grads2 = opt2.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                # logits.append(logit)
                tf.get_variable_scope().reuse_variables()
    # logits = tf.concat(0, logits)
    # accuracy = tower_acc(logits, labels_placeholder)
        accuracy = tower_acc(logit, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)
        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt1.apply_gradients(grads1)
        apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
        # null_op = tf.no_op()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(weights.values() + biases.values())
        init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
        if FLAGS.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            config.allow_soft_placement = True
        else:
            config = tf.ConfigProto()

        sess = tf.Session(
            config=config
        )
        sess.run(init)
        if os.path.isfile(model_filename) and use_pretrained_model:
            saver.restore(sess, model_filename)

        # Create summary writter
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)

        train_idxs, train_filenames, train_labels = input_data.get_list_of_filesnlabels('list/train.list')
        test_idxs, test_filenames, test_labels =input_data.get_list_of_filesnlabels('list/test.list')
        np_mean = np.load('./models/crop_mean.npy').reshape([c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, 3])


        train_file_start_position = 0
        test_file_start_position = 0
        ntrainbatches = int(len(train_filenames)/FLAGS.batch_size)
        ntestbatches = int(len(test_filenames)/FLAGS.batch_size)
        random.seed(FLAGS.randomseed)
        if FLAGS.max_steps is not 0:
            for step in xrange(FLAGS.max_steps):
                print '-'*32
                start_time = time.time()
                batch_id = step % (ntrainbatches)
                if batch_id == 0:
                    random.shuffle(train_idxs)
                    train_filenames = [train_filenames[i] for i in train_idxs]
                    train_labels = [train_labels[i] for i in train_idxs]
                start_idx = batch_id*FLAGS.batch_size
                end_idx = min((batch_id+1)*FLAGS.batch_size, len(train_filenames))
                batch_train_filenames = train_filenames[start_idx:end_idx]
                batch_train_labels = train_labels[start_idx:end_idx]

                assert len(batch_train_filenames)==len(batch_train_labels)
                tr_images, tr_labels= input_data.read_clip_and_label(
                    dirnames=batch_train_filenames,
                    labels= batch_train_labels,
                    np_mean= np_mean,
                    num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                    crop_size=c3d_model.CROP_SIZE,
                    RGB=FLAGS.rgb
                )

                sess.run(train_op, feed_dict={
                    images_placeholder: tr_images,
                    labels_placeholder: tr_labels
                })
                duration = time.time() - start_time
                print('Step {:d} \t time: {:.3f} sec, # of samples: {:d}'.format(step, duration,tr_labels.shape[0]))

                if (step+1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)

                    print('Test Data Eval:')
                    test_acc = performance_eval(sess, accuracy, images_placeholder, labels_placeholder,
                                                test_filenames, test_labels, FLAGS.batch_size, np_mean)
                    print ("Testing accuracy: " + "{:.5f}".format(test_acc))
        else:

            print('Train Data Eval:')
            train_acc = performance_eval(sess, accuracy, images_placeholder, labels_placeholder,
                                         train_filenames, train_labels, FLAGS.batch_size, np_mean)
            print ("Training accuracy: " + "{:.5f}".format(train_acc))

            print('Test Data Eval:')
            test_acc = performance_eval(sess, accuracy, images_placeholder, labels_placeholder,
                                        test_filenames, test_labels, FLAGS.batch_size, np_mean)
            print ("Testing accuracy: " + "{:.5f}".format(test_acc))
        print("done")


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
