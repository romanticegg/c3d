# a fully connected version of cifar 10
# this file creates inference, accuracy, loss, and training op
import tensorflow as tf
import utils
import os
import sys
import tarfile
import tf_easy_dir
from tf_utils import variable_with_weight_decay, variable_on_cpu, activation_summary
import cifar10_inputs as inputs
#todo: add batchnorm and VGG like 3*3 structure and batch normalization
NUM_CLASSES = 10

FLAGS=tf.app.flags.FLAGS

def _print_layer_info(layername, kernel=None, stride=None, reslt=None):

    print 'Layer {:s}'.format(layername)
    if kernel:
        print 'Kernel size [{:s}]'.format(', '.join(map(str, kernel)))
    if stride:
        print 'Stride size [{:s}]'.format(', '.join(map(str, stride)))
    if reslt:
        print 'Result size [{:s}]'.format(', '.join(map(str, reslt)))
    print '-' * 32


#fixme: batch normalization
BN_EPSILON = 0.001

def bn(x, isTraining=True, id_string=None, use_bias=False):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    if not id_string:
        id_string=""



    if use_bias:
        bias = variable_on_cpu('bias_{:s}'.format(id_string), params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias

    #todo: is this for all lay
    axis = list(range(len(x_shape) - 1))

    beta = variable_on_cpu('beta_{:s}'.format(id_string),
                           params_shape,
                           initializer=tf.zeros_initializer)
    gamma = variable_on_cpu('gamma_{:s}'.format(id_string),
                            params_shape,
                            initializer=tf.ones_initializer)

    moving_mean = variable_on_cpu('moving_mean_{:s}'.format(id_string),
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = variable_on_cpu('moving_variance_{:s}'.format(id_string),
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)

    if not isTraining:
        bn_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        bn_averages_op = bn_averages.apply([moving_mean, moving_variance])


        with tf.control_dependencies([bn_averages_op]):
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    else:
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    return x

#update: batch normalization added, so the training and validataion should differ
def inference(images, isTraining=True):
    print 'Model Initialization'
    print '*'*32
    #update: decompose the one layer into 2
    with tf.variable_scope('conv1') as scope:
        kernel1_1 = variable_with_weight_decay('weights1_1',
                                               shape=[3, 3, 3, 64],
                                               initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                               wd=0.0)
        # biases1_1 = variable_on_cpu('biases1_1', [64], initializer=tf.constant_initializer(0.0))

        conv1_1 = tf.nn.conv2d(images, kernel1_1, [1, 1, 1, 1], padding='SAME')
        conv1_1_bn = bn(conv1_1, isTraining=isTraining, id_string='1_1')
        intermediate1 = tf.nn.relu(conv1_1_bn, name='relu1_1')
        kernel1_2 = variable_with_weight_decay('weights1_2',
                                             shape=[3, 3, 64, 64],
                                             initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                             wd=0.0)
        # biases1_2 =variable_on_cpu('biases1_2', [64], initializer=tf.constant_initializer(0.0))

        conv1_2 = tf.nn.conv2d(intermediate1, kernel1_2, [1, 1, 1,1], padding='SAME')
        conv1_2_bn = bn(conv1_2, isTraining=isTraining, id_string='1_2')
        conv1 =tf.nn.relu(conv1_2_bn,name=scope.name)

        activation_summary(conv1)
        _print_layer_info('conv1_1', kernel=[3, 3, 3, 64], stride=[1, 1, 1, 1], reslt=intermediate1.get_shape().as_list())
        _print_layer_info('conv1_2', kernel=[3, 3, 64, 64], stride=[1, 1, 1, 1], reslt=conv1.get_shape().as_list())

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel2_1 = variable_with_weight_decay('weights2_1',
                                               shape=[3, 3, 64, 64],
                                               initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                               wd=0.0)
        # biases2_1 = variable_on_cpu('biases2_1', [64], tf.constant_initializer(0.0))
        conv2_1 = tf.nn.conv2d(norm1, kernel2_1, [1, 1, 1, 1], padding='SAME')
        conv2_1_bn = bn(conv2_1, isTraining=isTraining, id_string='2_1')

        intermediate2 =tf.nn.relu(conv2_1_bn, name='relu2_1')

        kernel2_2 = variable_with_weight_decay('weights2_2',
                                             shape=[3, 3, 64, 64],
                                             initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                             wd=0.0)
        # biases2_2 = variable_on_cpu('biases2_2', [64], tf.constant_initializer(0.0))
        conv2_2 = tf.nn.conv2d(intermediate2, kernel2_2, [1, 1, 1, 1], padding='SAME')
        conv2_2_bn = bn(conv2_2, isTraining=isTraining, id_string='2_2')

        conv2 =tf.nn.relu(conv2_2_bn, name='relu2_2')
        activation_summary(conv2)
        _print_layer_info('conv2', kernel=[3,3,64,64], stride=[1, 1, 1, 1], reslt=conv2.get_shape().as_list())

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # norm2
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(norm2, [FLAGS.batch_size, -1])
        # dim = reshape.get_shape()[1].value
        # norm2_shape = norm2.get_shape().as_list()
        weights = variable_with_weight_decay('weights', shape=[8, 8, 64, 384],
                                             initializer=tf.truncated_normal_initializer(stddev=0.04),
                                             wd=0.004)
        conv = tf.nn.conv2d(norm2, weights,[1, 1, 1, 1], padding='VALID')
        # biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        conv_bn = bn(conv, isTraining=isTraining, id_string='1')

        # pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(conv_bn, name=scope.name)
        activation_summary(conv3)
        _print_layer_info('conv3', kernel=[8, 8, 64, 384], stride=[1, 1, 1, 1], reslt=conv3.get_shape().as_list())

    #update: added a new normalization,  this lead to bad performance
    # norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # local4
    with tf.variable_scope('conv4') as scope:
        weights = variable_with_weight_decay('weights', shape=[1, 1, 384, 192],
                                             initializer=tf.truncated_normal_initializer(stddev=0.04), wd=0.004)
        conv = tf.nn.conv2d(conv3, weights,[1, 1, 1, 1], padding='VALID')
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv4)
        _print_layer_info('conv4', kernel=[1, 1, 384, 192], stride=[1, 1, 1, 1], reslt=conv4.get_shape().as_list())

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('classification') as scope:
        weights = variable_with_weight_decay('weights', [1, 1, 192, NUM_CLASSES],
                                             initializer=tf.truncated_normal_initializer(stddev=1/192.0), wd=0.0)
        biases = variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv4, weights, [1,1,1,1], padding='VALID')
        softmax = tf.nn.bias_add(conv, biases)
        _print_layer_info('classification', kernel=[1, 1, 192, NUM_CLASSES], stride=[1, 1, 1, 1],
                          reslt=softmax.get_shape().as_list())


        #todo: the following are used to deal with shapes with different sizes, to get mean
        softmax = tf.reduce_mean(softmax, axis=1, keep_dims=True)
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


def loss(logits, labels, isFinalLossOnly=False):
    labels=tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    if isFinalLossOnly:
        return cross_entropy_mean
    else:
        return tf.add_n(tf.get_collection('losses'), 'total_losses' )


#todo: adding moving averages to loss computation
def loss_moving_averages(total_loss):
    pass



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def train(total_loss, global_step, decay_every_n_step=None):
    if not decay_every_n_step:
        num_batches_per_epoch = inputs.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_every_n_step = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step=global_step,
                                    decay_steps=decay_every_n_step, decay_rate=LEARNING_RATE_DECAY_FACTOR)
    tf.summary.scalar('learning_rate', lr)

    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9)
    loss_list = tf.get_collection('losses')+[total_loss]  # a set of l2 loss, final entropy loss and the sum of all
    loss_averages_op = loss_averages.apply(loss_list)

    #todo: adding all the losses to summary:
    for s_loss in loss_list:
         tf.summary.scalar('{:s}(raw)'.format(s_loss.op.name), s_loss)
         tf.summary.scalar('{:s}(mv)'.format(s_loss.op.name), loss_averages.average(s_loss))

    with tf.control_dependencies([loss_averages_op]): # note this should be a list even if it is only one element
        opt =tf.train.GradientDescentOptimizer(learning_rate=lr)
        #todo: check the grads [grad, var]
        grads = opt.compute_gradients(total_loss)

    #note this is only an op without any output
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variable_average_op]):
        train_op=tf.no_op('train')

    return train_op

