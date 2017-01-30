import tensorflow as tf
import c3d_model_simple as c3d_model
import c3d_input_ucf101 as input_reader
import utils
import tf_easy_dir
import os
import glob
import tf_utils
import numpy as np
import sys

flags = tf.app.flags
flags.DEFINE_string('file_path', '/Users/zijwei/Dev/datasets/UCF-101-g16/train', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_integer('batch_size', 12, 'batch size[12]')
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
flags.DEFINE_integer('max_steps', 100000, 'Number of training steps[100000]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
flags.DEFINE_float('init_lr', 0.1, 'initial learning rate[0.05]')
flags.DEFINE_float('weight_decay_conv', 0.0, 'weight decay for convolutional layers [0.0]')
flags.DEFINE_float('weight_decay_fc', 0.004, 'weight decay for fully connected (fully convoluted) layers [0.004]')
FLAGS = flags.FLAGS

def main(argv=None):

    if len(glob.glob(os.path.join(FLAGS.file_path, '*.{:s}'.format(input_reader.TF_FORMAT)))) < 1:
        print "Check file path"
        return

    if not FLAGS.save_name:
        save_dir = os.path.join('c3dSave', utils.get_date_str())
    else:
        save_dir = os.path.join('c3dSave', FLAGS.save_name)

    save_locations = tf_easy_dir.tf_easy_dir(save_dir=save_dir)

    if FLAGS.rewrite:
        save_locations.clear_save_name()

    with tf.Graph().as_default() as graph:
        global_step =tf.get_variable(name='gstep', initializer=tf.constant(0), trainable=False)
        batch_images, batch_labels, batch_filenames = input_reader.inputs(FLAGS.file_path, isTraining=True)
        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32
        sys.stdout.flush()

        logits = c3d_model.inference_c3d(batch_images, isTraining=True)
        loss =c3d_model.loss(logits=logits, labels=batch_labels)
        train_op = c3d_model.train(loss, global_step)
        correct_ones = c3d_model.correct_ones(logits=logits, labels=batch_labels)

        saver = tf.train.Saver(max_to_keep=None)
        summary_op = tf.summary.merge_all()

        config = tf_utils.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            summary_writer = tf.summary.FileWriter(logdir=save_locations.summary_save_dir, graph=sess.graph)

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(FLAGS.max_steps):
                _, loss_, correct_ones_ = sess.run([train_op, loss, correct_ones])

                assert not np.isnan(loss_), 'Model diverged with loss = NaN, try again'

                # if (i+1) % 10 == 0:
                print '[{:s} -- {:08d}|{:08d}]\tloss : {:.3f}\t, correct ones [{:d}|{:d}]'.format(save_dir, i, FLAGS.max_steps,
                                                                                          loss_, correct_ones_, FLAGS.batch_size)
                sys.stdout.flush()
                if (i+1 % 100) ==0:
                    summary_ = sess.run(summary_op)
                    summary_writer.add_summary(summary_, global_step=global_step)

                if (i+1) % 2000 == 0:
                    save_path= os.path.join(save_locations.model_save_dir, 'model')
                    saver.save(sess=sess,global_step=global_step, save_path=save_path)
            coord.request_stop()
            coord.join(threads=threads)


if __name__ == '__main__':
    tf.app.run()

