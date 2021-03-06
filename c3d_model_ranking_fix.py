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

"""Builds the C3D with ranking network(Small) for UCF 101 dataset.

Fixing the size of the region ranking


"""

import tensorflow as tf
from tf_utils import variable_on_cpu, variable_with_weight_decay, bn, print_tensor_shape

FLAGS = tf.app.flags.FLAGS


# The UCF-101 dataset has 101 classes
NUM_CLASSES = 101

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 4.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.


# note:
#input: [batch, in_depth, in_height, in_width, in_channels]
#filter w: [filter_depth, filter_height, filter_width, in_channels, out_channels]
#strides:A list of ints that has length >= 5. 1-D tensor of length 5. The stride
# of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1

def region_ranking_3d_fix(tf_input, weight_shape, output_size, batch_size=None):
    """RegionRanking layer works similar to max-pooling

    Args:
        tf_input [batch_size, d, h, w, c]
        weight_shape [d, h, w, c, 1]  convolve with input to create weights for each component
        strids [h, w, c] shrink factor on each of the dimensions
        batch_size

    Returns:
        Output of the layer

    Rrainable variables:
        tf_ranking_w
    """
    # if not strides:
    # strides=[2, 2, 2]  # stride on depth, height, width

    # b_size, d, h, w, c = tf.shape(tf_input, tf.int32)
    batch_size = batch_size or FLAGS.batch_size
    n_outputs = reduce(lambda x, y: x * y, output_size)
    tf_input_shape = tf.shape(tf_input, out_type=tf.int32)

    tf_ranking_w = variable_with_weight_decay('w',
                                              shape=weight_shape,
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              wd=None)

    tf_weights = tf.nn.conv3d(tf_input, tf_ranking_w, strides=[1, 1, 1, 1, 1], padding='SAME')

    tf_input_line = tf.reshape(tf_input,
                               tf.pack([tf_input_shape[0], -1, tf_input_shape[-1]]))  # keep the batch_size and channels
    # fixme: it has to be in batch_size dimension because we want a sort in this table
    tf_weights_line = tf.reshape(tf_weights, tf.pack([tf_input_shape[0], -1]))

    # sorted=False enables that their spatial relationships will be roughly kept
    _, tf_indices = tf.nn.top_k(tf_weights_line, k=n_outputs,
                                sorted=False)

    tf_indices_line = tf.reshape(tf_indices, [-1])  # shape [batch_size, top_k_indices] to 1d vector
    tf_indices_helper = tf.expand_dims(tf.range(batch_size), 1)  # should be batch size
    tf_indices_helper = tf.tile(tf_indices_helper, multiples=[1, n_outputs])
    tf_indices_helper = tf.reshape(tf_indices_helper, [-1])

    tf_indices_2d = tf.stack([tf_indices_helper, tf_indices_line], axis=1)

    tf_input_shrinked = tf.gather_nd(tf_input_line, indices=tf_indices_2d)
    tf_output = tf.reshape(tf_input_shrinked,
                           tf.pack([tf_input_shape[0],  output_size[0],  output_size[1],
                                    output_size[2], tf_input_shape[4]]))
    return tf_output



def inference_c3d(inputs, isTraining=True):

    with tf.variable_scope('conv1') as scope:
        k1 = variable_with_weight_decay('w', shape=[3, 3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer(),
                                        wd=FLAGS.weight_decay_conv)
        conv1 = tf.nn.conv3d(inputs, k1, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')
        conv_bn1 = bn(conv1, isTraining=isTraining)
        conv_bn1 = tf.nn.relu(conv_bn1)
        print_tensor_shape(conv_bn1)

    with tf.variable_scope('pool1') as scope:
        pool1 = region_ranking_3d_fix(conv_bn1, [7, 7, 7, 64, 1], [16, 64, 64])
        print_tensor_shape(pool1)


    with tf.variable_scope('conv2') as scope:
        k2 = variable_with_weight_decay('w', shape=[3, 3, 3, 64, 128],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_conv)
        conv2 = tf.nn.conv3d(pool1, k2, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2')
        conv_bn2 = bn(conv2, isTraining=isTraining)
        conv_bn2 = tf.nn.relu(conv_bn2)
        print_tensor_shape(conv_bn2)

    pool2 = tf.nn.max_pool3d(conv_bn2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')
    print_tensor_shape(pool2)

    with tf.variable_scope('conv3') as scope:
        k3 = variable_with_weight_decay('w', shape=[3, 3, 3, 128, 256],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_conv)
        conv3 = tf.nn.conv3d(pool2, k3, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3')
        conv_bn3 = bn(conv3, isTraining=isTraining)
        conv_bn3 = tf.nn.relu(conv_bn3)
        print_tensor_shape(conv_bn3)

    pool3 = tf.nn.max_pool3d(conv_bn3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')
    print_tensor_shape(pool3)

    with tf.variable_scope('conv4') as scope:
        k4 = variable_with_weight_decay('w', shape=[3, 3, 3, 256, 256],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_conv)
        conv4 = tf.nn.conv3d(pool3, k4, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4')
        conv_bn4 = bn(conv4, isTraining=isTraining)
        conv_bn4 =tf.nn.relu(conv_bn4)
        print_tensor_shape(conv_bn4)

    pool4 = tf.nn.max_pool3d(conv_bn4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')
    print_tensor_shape(pool4)

    with tf.variable_scope('conv5') as scope:
        k5 = variable_with_weight_decay('w', shape=[3, 3, 3, 256, 256],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_conv)
        conv5 = tf.nn.conv3d(pool4, k5, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5')
        conv_bn5 = bn(conv5, isTraining=isTraining)
        conv_bn5 = tf.nn.relu(conv_bn5)
        print_tensor_shape(conv_bn5)

    pool5 = tf.nn.max_pool3d(conv_bn5, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool5')
    print_tensor_shape(pool5)

    with tf.variable_scope('fc1') as scope:
        kfc1 = variable_with_weight_decay('w', shape=[1, 4, 4, 256, 2048],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_fc)
        conv_fc1 = tf.nn.conv3d(pool5, kfc1, strides=[1, 1, 1, 1, 1], padding='VALID', name='fc1')
        conv_bn_fc1 = bn(conv_fc1, isTraining=isTraining)
        print_tensor_shape(conv_bn_fc1)

    if isTraining:
        conv_bn_fc1 = tf.nn.dropout(conv_bn_fc1, FLAGS.dropout)

    with tf.variable_scope('fc2') as scope:
        kfc2 = variable_with_weight_decay('w', shape=[1, 1, 1, 2048, 2048],
                                        initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_fc)
        conv_fc2 = tf.nn.conv3d(conv_bn_fc1, kfc2, strides=[1, 1, 1, 1, 1], padding='VALID', name='fc2')
        conv_bn_fc2 = bn(conv_fc2, isTraining=isTraining)
        print_tensor_shape(conv_bn_fc2)

    if isTraining:
        conv_bn_fc2 = tf.nn.dropout(conv_bn_fc2, FLAGS.dropout)

    with tf.variable_scope('classification') as scope:
        weights = variable_with_weight_decay('w', [1, 1, 1, 2048, NUM_CLASSES],
                                             initializer=tf.contrib.layers.xavier_initializer(), wd=FLAGS.weight_decay_fc)
        biases = variable_on_cpu('b', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv_bn_fc2, weights, strides=[1, 1, 1, 1, 1], padding='VALID')
        softmax = tf.nn.bias_add(conv, biases)
        print_tensor_shape(softmax, 'softmax-before')


        #todo: the following are used to deal with shapes with different sizes, to get mean
        softmax = tf.reduce_mean(softmax, axis=1, keep_dims=True)
        softmax = tf.reduce_mean(softmax, axis=2, keep_dims=True)
        softmax = tf.reduce_mean(softmax, axis=3, keep_dims=True)

        softmax = tf.squeeze(softmax, axis=[1, 2, 3])
        print_tensor_shape(softmax, 'softmax-after')

    # Output: class prediction
    return softmax


def correct_ones(logits, labels, k=1):
    correct_prediction = tf.nn.in_top_k(logits, labels, k)
    correct_prediction = tf.cast(correct_prediction, tf.int32)
    # pred = tf.cast(tf.nn.in_top_k(logits, labels, k=1), tf.float32)
    n_corrects = tf.reduce_sum(correct_prediction,name='correct_n')
    # tf.summary.scalar('correct_ones', n_corrects)
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


def train(total_loss, global_step, decay_every_n_step):
    # if not decay_every_n_step:
    #     num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    #     decay_every_n_step = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.init_lr, global_step=global_step,
                                    decay_steps=decay_every_n_step, decay_rate=FLAGS.lr_decay_rate)
    tf.summary.scalar('learning_rate', lr)

    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9)
    loss_list = tf.get_collection('losses')+[total_loss]  # a set of l2 loss, final entropy loss and the sum of all
    loss_averages_op = loss_averages.apply(loss_list)

    # todo: adding all the losses to summary:
    # for s_loss in loss_list:
    #      tf.summary.scalar('{:s}(raw)'.format(s_loss.op.name), s_loss)
    #      tf.summary.scalar('{:s}(mv)'.format(s_loss.op.name), loss_averages.average(s_loss))

    with tf.control_dependencies([loss_averages_op]): # note this should be a list even if it is only one element
        opt =tf.train.GradientDescentOptimizer(learning_rate=lr)
        #todo: check the grads [grad, var]
        grads = opt.compute_gradients(total_loss)

    # note this is only an op without any output
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variable_average_op]):
        train_op = tf.no_op('train')

    return train_op, lr
