# a fully connected version of cifar 10
import tensorflow as tf
import utils
import os
import sys
from six.moves import urllib
import tarfile
import tf_easy_dir
NUM_CLASSES = 10






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


class model:

    def inference(images):
        with tf.variable_scope('conv1') as scope:
            pass


def train():
    # create a model:
    with tf.Graph().as_default():
        global_steps = tf.Variable(name='gstep', initial_value= 0, trainable=False)
        images = tf.Variable(tf.ones(shape=[10,32,32,3]), dtype=tf.float32, name='input')
        # images = tf.placeholder(dtype=tf.float32, shape=None, name='input_images')
        # labels = tf.placeholder(dtype=tf.int32, shape=None, name= 'labels')
        print 'size of input: [{:s}]'.format(', '.join(map(str, images.get_shape().as_list())))

        # batch_sz = tf.shape(images)
        # tf.Print(batch_sz, [batch_sz],)
        batch_sz = images.get_shape().as_list()[0] # current batch size?
        # inference part:
        # first layer
        with tf.variable_scope('conv1') as scope:
            w1 = _variable_with_weight_decay('w', [5,5,3,64], stddev=5e-2, wd=0)
            b1 = _variable_on_cpu('b',[64],tf.constant_initializer(0))
            pre_activation1 = tf.nn.bias_add(tf.nn.conv2d(images, w1, [1,1,1,1],padding="SAME"), b1)
            conv1 = tf.nn.relu(pre_activation1, 'relu')

        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

        with tf.variable_scope('conv2') as scope:
            w2 = _variable_with_weight_decay('w', [5,5,64,64], stddev=5e-2, wd=0)
            b2 = _variable_on_cpu('b',[64],tf.constant_initializer(0))
            pre_activation2 = tf.nn.bias_add(tf.nn.conv2d(norm1, w2, [1,1,1,1],padding="SAME"), b2)
            conv2 = tf.nn.relu(pre_activation2, 'relu')

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

        with tf.variable_scope('conv4') as scope:
            # reshape = tf.reshape(norm2, [nimages, -1])
            # dim = reshape.get_shape().as_list()[1]
            w4 = _variable_with_weight_decay('w', [1, 1, 384, 192], stddev=0.04, wd=0.004)
            b4 = _variable_on_cpu('b',[192],tf.constant_initializer(0))
            pre_activation4 = tf.nn.bias_add(tf.nn.conv2d(full3, w4, [1,1,1,1], padding='VALID'), b4)
            full4 = tf.nn.relu(pre_activation4, 'relu')

        with tf.variable_scope('class') as scope:
            # reshape = tf.reshape(norm2, [batch_sz, -1])
            # dim = reshape.get_shape().as_list()[1]
            w_o = _variable_with_weight_decay('w', [1, 1, 192, NUM_CLASSES], stddev=0.04, wd=0.004)
            b_o = _variable_on_cpu('b',[NUM_CLASSES],tf.constant_initializer(0))
            softmax = tf.nn.bias_add(tf.nn.conv2d(full4, w_o, [1,1,1,1], padding='VALID'), b_o)
            # full3 = tf.nn.relu(pre_activation1, 'relu')

        # up-sample softmax layer and fuse with pool2 layer
        with tf.variable_scope('upscale_1') as scope:
             pool2_size = pool2.get_shape().as_list()

             upscale2_shape =tf.stack([pool2_size[0], pool2_size[1], pool2_size[2], NUM_CLASSES])
             w_upscale = _variable_with_weight_decay('w', [8,8, NUM_CLASSES, NUM_CLASSES], stddev=5e2, wd=0.004)
             #fixme: problem here, check the dimension problem here!
             upscale2 = tf.nn.conv2d_transpose(softmax, w_upscale, upscale2_shape, [1, 1, 1, 1], padding='VALID')
             #todo: not finished yet!
             pool2_upscale = _score_layer(pool2,'pool2_transpose', NUM_CLASSES)
             upscale2 = tf.add(upscale2, pool2_upscale)

        with tf.variable_scope('upscale_2') as scope:
             pool1_size = pool1.get_shape().as_list()

             upscale1_shape =tf.stack([pool1_size[0], pool1_size[1], pool1_size[2], NUM_CLASSES])
             w_upscale = _variable_with_weight_decay('w', [4,4, NUM_CLASSES, NUM_CLASSES], stddev=5e2, wd=0.004)
             upscale1 = tf.nn.conv2d_transpose(upscale2, w_upscale, upscale1_shape, [1, 2, 2, 1], padding='SAME')
             #todo: not finished yet!
             pool1_upscale = _score_layer(pool1,'pool1_transpose', NUM_CLASSES)
             upscale1 = tf.add(upscale1, pool1_upscale)

        with tf.variable_scope('final') as scope:
        #final: up to final image size:
            image_sz = images.get_shape().as_list()
            final_shape = tf.stack([image_sz[0], image_sz[1], image_sz[2], NUM_CLASSES])
            w_upscale = _variable_with_weight_decay('w', [4, 4, NUM_CLASSES, NUM_CLASSES], stddev=5e2, wd=0.004)
            pred = tf.nn.conv2d_transpose(upscale1, w_upscale, final_shape, [1, 2, 2, 1], padding='SAME')

        # classification preidiction results
        # pred = tf.argmax(softmax, axis=3)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(4):
                o_, pred_ = sess.run([softmax, pred])

                print '{:d}: shape-output:[{:s}], shape-pred: [{:s}]'.format(i, ', '.join(map(str, o_.shape)), ', '.join(map(str, pred_.shape)) )

            # output = sess.run(softmax,feed_dict={images:})

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


#no need
def download(save_dir, rewrite=False):
    save_dir = utils.get_dir()
    if rewrite:
        utils.clear_dir(save_dir)

    filename = DATA_URL.split(os.pathsep)[-1]
    filepath = os.path.join(save_dir, filename)
    if not os.path.isfile(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(save_dir)


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
FLAGS = flags.FLAGS



def main(argv = None):
    if not FLAGS.save_name:
        save_dir = utils.get_date_str()
    else:
        save_dir = FLAGS.save_name

    save_locations = tf_easy_dir.tf_easy_dir(save_dir=save_dir)
    if FLAGS.rewrite:
        save_locations.clear_save_name()




if __name__ == '__main__':
    tf.app.run()





