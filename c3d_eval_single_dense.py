import tensorflow as tf
import c3d_model_simple as c3d_model
import tf_easy_dir
import utils
import tf_utils
import c3d_input_ucf101 as c3d_inputs
import os
import numpy as np
import math
import progressbar
import glob

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string('model', None, 'the model to evaluate on')
flags.DEFINE_integer('batch_size', 1, 'batch size[1]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16[False].""")
FLAGS = flags.FLAGS


#todo: add multiple GPU execution

def eval():

    NUM_EXAMPLES_FOR_EVAL = len(glob.glob(os.path.join(FLAGS.data_dir, '*.{:s}'.format(c3d_inputs.TF_FORMAT))))
    with tf.Graph().as_default() as graph:
        [batch_images, batch_labels,_] = c3d_inputs.inputs(FLAGS.data_dir, isTraining=False)

        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32

        logits = c3d_model.inference_c3d(batch_images, isTraining=False)
        correct_ones = c3d_model.correct_ones(logits=logits, labels=batch_labels)


        variable_averages = tf.train.ExponentialMovingAverage(
            c3d_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver()

        config = tf_utils.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            saver.restore(sess=sess,save_path=FLAGS.model)
            nbatches = int(math.ceil(NUM_EXAMPLES_FOR_EVAL*1.0/FLAGS.batch_size))

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