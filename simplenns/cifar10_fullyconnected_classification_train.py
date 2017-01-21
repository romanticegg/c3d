# a fully connected version of cifar 10
import tensorflow as tf
import utils
import tf_utils
import os
import sys
import tarfile
sys.path.append('../')
import tf_easy_dir
NUM_CLASSES = 10
import cifar10_inputs
import numpy as np


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd=None):
    var = _variable_on_cpu(name=name, shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev))

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

# def conv2d

def _up_score_layer(input, shape, name, num_class = NUM_CLASSES, ksize=4, kstride=2):
    stride=[1, kstride, kstride, 1]
    #todo : not done yet
    with tf.variable_scope(name) as scope:
        pass


def _score_layer(bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape().as_list()[3]
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        # if name == "score_fr":
        #     num_input = in_features
        #     stddev = (2 / num_input) ** 0.5
        # elif name == "score_pool4":
        #     stddev = 0.001
        # elif name == "score_pool3":
        #     stddev = 0.0001
        # Apply convolution
        # w_decay = self.wd

        weights = _variable_with_weight_decay(name='w', shape=shape, stddev=5e-2, wd=0.004)

        # Apply bias
        conv_biases = _variable_on_cpu('b',[num_classes], tf.constant_initializer(0))
        activation = tf.nn.bias_add(tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME'), conv_biases)
        # _activation_summary(bias)
        return activation





def train():
    # create a model:
    with tf.Graph().as_default():
        print 'Graph initialization'
        global_steps = tf.Variable(name='gstep', initial_value= 0, trainable=False)
        [batch_images, batch_labels] = cifar10_inputs.inputs(FLAGS.data_dir, FLAGS.batch_size, isTraining=True, isRandom=False)

        # images = tf.Variable(tf.ones(shape=[10,32,32,3]), dtype=tf.float32, name='input')
        # images = tf.placeholder(dtype=tf.float32, shape=None, name='input_images')
        # labels = tf.placeholder(dtype=tf.int32, shape=None, name= 'labels')
        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32
        # batch_sz = tf.shape(images)
        # tf.Print(batch_sz, [batch_sz],)
        # batch_sz = images.get_shape().as_list()[0] # current batch size?
        # inference part:
        # first layer
        with tf.variable_scope('conv1') as scope:
            w1 = _variable_with_weight_decay('w', [5,5,3,64], stddev=5e-2, wd=0)
            b1 = _variable_on_cpu('b',[64],tf.constant_initializer(0))
            pre_activation1 = tf.nn.bias_add(tf.nn.conv2d(batch_images, w1, [1,1,1,1],padding="SAME"), b1)
            conv1 = tf.nn.relu(pre_activation1, 'relu')
            print 'size of first layer output: [{:s}]'.format(', '.join(map(str, conv1.get_shape().as_list())))

        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
        print 'size of first pool layer output: [{:s}]'.format(', '.join(map(str, norm1.get_shape().as_list())))

        with tf.variable_scope('conv2') as scope:
            w2 = _variable_with_weight_decay('w', [5,5,64,64], stddev=5e-2, wd=0)
            b2 = _variable_on_cpu('b',[64],tf.constant_initializer(0))
            pre_activation2 = tf.nn.bias_add(tf.nn.conv2d(norm1, w2, [1,1,1,1],padding="SAME"), b2)
            conv2 = tf.nn.relu(pre_activation2, 'relu')
            print 'size of conv2 output: [{:s}]'.format(', '.join(map(str, conv2.get_shape().as_list())))

        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        norm2 = tf.nn.lrn(pool2, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

        print 'Size of norm2 is [{:s}]'.format(', '.join(map(str, norm2.get_shape().as_list())))

        with tf.variable_scope('conv3') as scope:
            # reshape = tf.reshape(norm2, [-1, 8*8*64])
            # if input is [batch, 32, 32, 3], current norm2 is [batch_size, 8, 8, 64]
            # dim = reshape.get_shape().as_list()[1]
            w3 = _variable_with_weight_decay('w', [8, 8, 64, 384], stddev=0.04, wd=0.004)
            b3 = _variable_on_cpu('b',[384],tf.constant_initializer(0))
            pre_activation3 = tf.nn.bias_add(tf.nn.conv2d(norm2, w3, [1,1,1,1], padding='VALID'), b3)
            full3 = tf.nn.relu(pre_activation3, 'relu')
            print 'Size of conv3 is [{:s}]'.format(', '.join(map(str, full3.get_shape().as_list())))

        with tf.variable_scope('conv4') as scope:
            # reshape = tf.reshape(norm2, [nimages, -1])
            # dim = reshape.get_shape().as_list()[1]
            w4 = _variable_with_weight_decay('w', [1, 1, 384, 192], stddev=0.04, wd=0.004)
            b4 = _variable_on_cpu('b',[192],tf.constant_initializer(0))
            pre_activation4 = tf.nn.bias_add(tf.nn.conv2d(full3, w4, [1,1,1,1], padding='VALID'), b4)
            full4 = tf.nn.relu(pre_activation4, 'relu')
            print 'Size of conv4 is [{:s}]'.format(', '.join(map(str, full4.get_shape().as_list())))

        with tf.variable_scope('classification') as scope:
            # reshape = tf.reshape(norm2, [batch_sz, -1])
            # dim = reshape.get_shape().as_list()[1]
            w_o = _variable_with_weight_decay('w', [1, 1, 192, NUM_CLASSES], stddev=0.04, wd=0.004)
            b_o = _variable_on_cpu('b',[NUM_CLASSES],tf.constant_initializer(0))
            softmax = tf.nn.bias_add(tf.nn.conv2d(full4, w_o, [1,1,1,1], padding='VALID'), b_o)
            print 'Size of final layer is [{:s}]'.format(', '.join(map(str, softmax.get_shape().as_list())))

            softmax = tf.reduce_mean(softmax, axis=1, keep_dims=True)
            softmax = tf.reduce_mean(softmax, axis=2, keep_dims=True)
            softmax = tf.squeeze(softmax,axis=[1,2])
            print 'Size of reduced final layer is [{:s}]'.format(', '.join(map(str, softmax.get_shape().as_list())))

        with tf.variable_scope('loss') as scope:
            batch_labels = tf.cast(batch_labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            softmax, batch_labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)


            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            # classification preidiction results
            # pred = tf.argmax(softmax, axis=3)

        with tf.variable_scope('optimization') as scope:

            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
            LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
            INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

            num_batches_per_epoch = cifar10_inputs. NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            global_steps,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            # tf.scalar_summary('learning_rate', lr)

            # Generate moving averages of all losses and associated summaries.
            # loss_averages_op = _add_loss_summaries(total_loss)

            # Compute gradients.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            losses = tf.get_collection('losses')

            loss_averages_op = loss_averages.apply(losses + [total_loss])
            with tf.control_dependencies([loss_averages_op]):
                opt = tf.train.GradientDescentOptimizer(lr)
                grads = opt.compute_gradients(total_loss)

            # Apply gradients.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_steps)



            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_steps)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                train_op = tf.no_op(name='train')

            top_k = tf.nn.in_top_k(softmax, batch_labels ,1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(4):
                _, top_k_, = sess.run([train_op, top_k])
                print '{:d}: precision:[{:d} / {:d}]'.format(i, np.sum(top_k_), FLAGS.batch_size)

                # print '{:d}: shape-output:[{:s}], shape-pred: [{:s}]'.format(i, ', '.join(map(str, o_.shape)), ', '.join(map(str, pred_.shape)) )
                # print '{:d}: shape-output:[{:s}]'.format(i, ', '.join(map(str, o_.shape)))

            # output = sess.run(softmax,feed_dict={images:})

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


#fixme: no need right now.
# def download(save_dir, rewrite=False):
#     save_dir = utils.get_dir()
#     if rewrite:
#         utils.clear_dir(save_dir)
#
#     filename = DATA_URL.split(os.pathsep)[-1]
#     filepath = os.path.join(save_dir, filename)
#     if not os.path.isfile(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
#                                                              float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()
#
#         filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#
#     tarfile.open(filepath, 'r:gz').extractall(save_dir)


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
flags.DEFINE_integer('batch_size', 10, 'training batch size [10]')
flags.DEFINE_integer('max_steps', 5000, 'The max steps of learning')
FLAGS = flags.FLAGS



# def main(argv = None):
#     if not FLAGS.save_name:
#         save_dir = utils.get_date_str()
#     else:
#         save_dir = FLAGS.save_name
#
#     save_locations = tf_easy_dir.tf_easy_dir(save_dir=save_dir)
#     if FLAGS.rewrite:
#         save_locations.clear_save_name()

def main(argv = None):
    train()



if __name__ == '__main__':
    tf.app.run()





