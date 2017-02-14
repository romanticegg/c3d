import tensorflow as tf
import tf_easy_dir
import utils
import tf_utils_inner
import cifar10_inputs
import os
import numpy as np
import importlib
import sys

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_integer('batch_size', 128, 'batch size[128]')
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
flags.DEFINE_integer('max_steps', 5000, 'Number of training steps[5000]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16[False].""")
flags.DEFINE_string('architecture', 'cifar10_simple_baseline', 'version of model to use')
flags.DEFINE_float('lr', 0.1, 'learning_rate[0.1]')
flags.DEFINE_integer('epochs_per_decay', 10, 'number of epochs per decay [10]')
FLAGS = flags.FLAGS


#todo: add multiple GPU execution

def train():
    # import cifar10_simple_baseline as cifar10_model

    cifar10_model = importlib.import_module(FLAGS.architecture)
    if not FLAGS.save_name:
        save_dir = os.path.join('Save', utils.get_date_str()+FLAGS.architecture)
    else:
        save_dir = os.path.join('Save', FLAGS.save_name)

    save_locations = tf_easy_dir.tf_easy_dir(save_dir=save_dir)
    if FLAGS.rewrite:
        save_locations.clear_save_name()

    with tf.Graph().as_default() as graph:
        global_step =tf.get_variable(name='gstep', initializer=tf.constant(0), trainable=False)
        [batch_images, batch_labels] = cifar10_inputs.inputs(FLAGS.data_dir, FLAGS.batch_size, isTraining=True, isRandom=True)
        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32
        sys.stdout.flush()

        logits = cifar10_model.inference(batch_images, isTraining=True)
        loss =cifar10_model.loss(logits=logits, labels=batch_labels)
        train_op, lr = cifar10_model.train(loss, global_step)
        update_ops = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        correct_ones = cifar10_model.correct_ones(logits=logits, labels=batch_labels)

        # debug:
        with tf.variable_scope("", reuse=True):
            moving_mean1 = tf.get_variable('inference/Conv/BatchNorm/moving_mean')
            moving_variance1 = tf.get_variable('inference/Conv/BatchNorm/moving_variance')


        saver = tf.train.Saver(max_to_keep=None)
        summary_op = tf.summary.merge_all()

        config = tf_utils_inner.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            summary_writer = tf.summary.FileWriter(logdir=save_locations.summary_save_dir, graph=sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(FLAGS.max_steps):
                _, _,loss_, correct_ones_, lr_ = sess.run([train_op, update_ops,loss, correct_ones, lr])

                assert not np.isnan(loss_), 'Model diverged with loss = NaN, try again'

                # if (i+1) % 10 == 0:
                print '[{:s} -- {:08d}|{:08d}]\tloss : {:.3f}\t, l-rate: {:.6f}\tcorrect ones [{:d}|{:d}]'.format(save_dir, i, FLAGS.max_steps,
                                                                                          loss_, lr_, correct_ones_, FLAGS.batch_size)
                sys.stdout.flush()

                mm1, mv1 = sess.run([moving_mean1, moving_variance1])
                print 'Sum of moving mean: {:.6f} \t, moving variance: {:.06f}'.format(np.sum(mm1), np.sum(mv1))

                if (i+1 % 100) == 0:
                    summary_ = sess.run(summary_op)
                    summary_writer.add_summary(summary_, global_step=global_step)

                if (i+1) % 2000 == 0:
                    save_path= os.path.join(save_locations.model_save_dir, 'model')
                    saver.save(sess=sess,global_step=global_step, save_path=save_path)
            coord.request_stop()
            coord.join(threads=threads)

def main(argv=None):
    train()



if __name__ == '__main__':
    tf.app.run()