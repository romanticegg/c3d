# a fully connected version of cifar 10
# this file creates inference, accuracy, loss, and training op
# using abstractions on CNN
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import utils
import os
import sys
import tarfile
import tf_easy_dir
from tf_utils_inner import variable_with_weight_decay, variable_on_cpu, activation_summary
import cifar10_inputs as inputs
#todo: add batchnorm and VGG like 3*3 structure and batch normalization
NUM_CLASSES = 10

FLAGS=tf.app.flags.FLAGS


def batch_norm_decorator(isTraining=True):
    # return tf.nn.relu(batch_norm_func)
    def warpper(x):
        return tf.nn.relu(tcl.batch_norm(x, decay=0.9, center=True, scale=True, is_training=isTraining))
    return warpper

# def relu_batch_norm(x):
#     return tf.nn.relu(tcl.batch_norm(x))

#update: batch normalization added, so the training and validataion should differ
def inference(images, isTraining=True):
    print 'Model Initialization'
    print '*'*32
    #update: decompose the one layer into 2
    with tf.variable_scope('inference') as scope:
        conv11 = tcl.conv2d(images, 64, [3, 3], stride=1, activation_fn=batch_norm_decorator(isTraining=isTraining))
        conv12 = tcl.conv2d(conv11, 64, [3, 3], stride=1, activation_fn=batch_norm_decorator(isTraining=isTraining))
        pool1 = tf.nn.max_pool(conv12, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

        conv21 = tcl.conv2d(pool1, 128, [3, 3], stride=1, activation_fn=batch_norm_decorator(isTraining=isTraining))
        conv22 = tcl.conv2d(conv21, 128, [3, 3], stride=1, activation_fn=batch_norm_decorator(isTraining=isTraining))
        pool2 = tf.nn.max_pool(conv22, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        conv3 = tcl.conv2d(pool2, 384, [8, 8], stride=1, padding='VALID', activation_fn=batch_norm_decorator(isTraining=isTraining),variables_collections=['loss'])
        conv4 = tcl.conv2d(conv3, 192, [1, 1], stride=1, padding='VALID', activation_fn=batch_norm_decorator(isTraining=isTraining), variables_collections=['loss'])

        final = tcl.conv2d(conv4, NUM_CLASSES, [1, 1], stride=1, padding='VALID', activation_fn=tf.identity)

        softmax = tf.reduce_mean(final, axis=1, keep_dims=True)
        softmax = tf.reduce_mean(softmax, axis=2, keep_dims=True)
        softmax = tf.squeeze(softmax, axis=[1, 2])

        activation_summary(softmax)

    return softmax


def correct_ones(logits, labels, k=1):
    correct_prediction = tf.nn.in_top_k(logits, labels, k)
    correct_prediction = tf.cast(correct_prediction, tf.int32)
    # pred = tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32)
    n_corrects = tf.reduce_sum(correct_prediction,name='correct_n')
    tf.summary.scalar('correct_ones', n_corrects)
    return n_corrects


# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 1     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
# WEIGHT_DECAY = 0.0005

def loss(logits, labels, isFinalLossOnly=False):
    labels=tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    reg = tcl.apply_regularization(
        tcl.l2_regularizer(FLAGS.layer_weight_decay),
        weights_list=[var for var in tf.get_collection('loss') if 'weights' in var.name]
    )

    return cross_entropy_mean + reg









def train(total_loss, global_step, decay_every_n_step=None):
    if not decay_every_n_step:
        num_batches_per_epoch = inputs.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_every_n_step = int(num_batches_per_epoch * FLAGS.epochs_per_decay)

    print 'Decay every {:d} steps'.format(decay_every_n_step)
    sys.stdout.flush()
    lr = tf.train.exponential_decay(FLAGS.lr, global_step=global_step,
                                    decay_steps=decay_every_n_step, decay_rate=LEARNING_RATE_DECAY_FACTOR)
    tf.summary.scalar('learning_rate', lr)

    # loss_averages = tf.train.ExponentialMovingAverage(decay=0.9)
    # loss_list = tf.get_collection('losses')+[total_loss]  # a set of l2 loss, final entropy loss and the sum of all
    # loss_averages_op = loss_averages.apply(loss_list)

    # #todo: adding all the losses to summary:
    # for s_loss in loss_list:
    #      tf.summary.scalar('{:s}(raw)'.format(s_loss.op.name), s_loss)
    #      tf.summary.scalar('{:s}(mv)'.format(s_loss.op.name), loss_averages.average(s_loss))

    # with tf.control_dependencies([loss_averages_op]): # note this should be a list even if it is only one element
    train_op =tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, global_step=global_step)
    #     #todo: check the grads [grad, var]
    #     grads = opt.compute_gradients(total_loss)

    # #note this is only an op without any output
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #
    # variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    # variable_average_op = variable_average.apply(tf.trainable_variables())
    # with tf.control_dependencies([apply_gradient_op, variable_average_op]):
    #     train_op=tf.no_op('train')

    return train_op, lr

