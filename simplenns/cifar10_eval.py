import tensorflow as tf
import cifar10_fc
import tf_easy_dir
import utils
import tf_utils
import cifar10_inputs
import os
import numpy as np
import math

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string('model', None, 'the model to evaluate on')
flags.DEFINE_integer('batch_size', 50, 'batch size[50]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
FLAGS = flags.FLAGS


#todo: add multiple GPU execution

def eval():

    with tf.Graph().as_default() as graph:
        [batch_images, batch_labels] = cifar10_inputs.inputs(FLAGS.data_dir, FLAGS.batch_size, isTraining=False, isRandom=False)

        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32

        logits = cifar10_fc.inference(batch_images)
        correct_ones = cifar10_fc.correct_ones(logits=logits, labels=batch_labels)
        saver = tf.train.Saver()

        config = tf_utils.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            saver.restore(sess=sess,save_path=FLAGS.model)
            nbatches = math.ceil(cifar10_inputs.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL*1.0/FLAGS.batch_size)

            tf.train.start_queue_runners(sess=sess)
            a_correct = 0
            a_tested = 0
            for i in xrange(nbatches):
                correct_ones_ = sess.run(correct_ones)
                print 'Batch: {:d}|{:d} \t correct: {:d}'.format(i, nbatches, correct_ones)
                a_correct += correct_ones_
                a_tested  += FLAGS.batch_size


            print 'Done evaluation, [{:d} out of {:d}], rate: {:.3f}'.format(a_correct, a_tested, a_correct*1.0/a_tested)



def main(argv=None):
    if not FLAGS.model:
        'Please indicate the model path'
        return
    eval()


if __name__ == '__main__':
    tf.app.run()