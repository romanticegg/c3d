import tensorflow as tf
# import cifar10_baseline as cifar10_model
import tf_easy_dir
import utils
import tf_utils_inner
import cifar10_inputs
import os
import numpy as np
import math
import progressbar
import importlib


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string('model', None, 'the model to evaluate on')
flags.DEFINE_integer('batch_size', 50, 'batch size[50]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16[False].""")
flags.DEFINE_string('architecture', 'cifar10_simple_baseline', 'version of model to use')

FLAGS = flags.FLAGS


#todo: add multiple GPU execution

def eval():
    cifar10_model = importlib.import_module(FLAGS.architecture)

    with tf.Graph().as_default() as graph:
        [batch_images, batch_labels] = cifar10_inputs.inputs(FLAGS.data_dir, FLAGS.batch_size, isTraining=False, isRandom=False )

        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32

        logits = cifar10_model.inference(batch_images, isTraining=True)
        correct_ones = cifar10_model.correct_ones(logits=logits, labels=batch_labels)


        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10_model.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()

        config = tf_utils_inner.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            # sess.run(tf.variables_initializer(tf.global_variables()))

            saver.restore(sess=sess,save_path=FLAGS.model)
            nbatches = int(math.ceil(cifar10_inputs.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL*1.0/FLAGS.batch_size))
            #debug:
            # with tf.variable_scope("", reuse=True):
            #     moving_mean1 = tf.get_variable('inference/Conv/BatchNorm/moving_mean')
            #     moving_variance1 = tf.get_variable('inference/Conv/BatchNorm/moving_variance')

            #todo: remember the pattern
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            a_correct = 0
            a_tested = 0
            pbar =progressbar.ProgressBar()

            for i in pbar(range(nbatches)):
                correct_ones_ = sess.run(correct_ones)
                # print 'Batch: {:d}|{:d} \t correct: {:d}'.format(i, nbatches, correct_ones_)
                a_correct += correct_ones_
                a_tested += FLAGS.batch_size

            coord.request_stop()
            coord.join(threads=threads)

            # print 'total_sample_count:\t{:d}, \t correct ones:\t{:d}'.format(total_sample_count, true_count)
            print 'Done evaluation, [{:d} out of {:d}], rate: {:.3f}'.format(a_correct, a_tested, a_correct*1.0/a_tested)

def main(argv=None):
    if not FLAGS.model:
        print 'Please indicate the model path'
        return
    eval()


if __name__ == '__main__':
    tf.app.run()