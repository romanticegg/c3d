import tensorflow as tf
import cifar10_fc
import tf_easy_dir
import utils
import tf_utils
import cifar10_inputs
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_integer('batch_size', 64, 'batch size[64]')
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
flags.DEFINE_integer('max_steps', 5000, 'Number of training steps[5000]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
FLAGS = flags.FLAGS


#todo: add multiple GPU execution

def train():
    if not FLAGS.save_name:
        save_dir = os.path.join('Save', utils.get_date_str())
    else:
        save_dir = os.path.join('Save', FLAGS.save_name)

    save_locations = tf_easy_dir.tf_easy_dir(save_dir=save_dir)
    if FLAGS.rewrite:
        save_locations.clear_save_name()

    with tf.Graph().as_default() as graph:
        global_step =tf.get_variable(name='gstep', initializer=tf.constant(0), trainable=False)
        [batch_images, batch_labels] = cifar10_inputs.inputs(FLAGS.data_dir, FLAGS.batch_size, isTraining=True, isRandom=False)

        print 'size of image input: [{:s}]'.format(', '.join(map(str, batch_images.get_shape().as_list())))
        print 'size of labels : [{:s}]'.format(', '.join(map(str, batch_labels.get_shape().as_list())))
        print '-'*32

        logits = cifar10_fc.inference(batch_images)
        loss =cifar10_fc.loss(logits=logits, labels=batch_labels)
        train_op = cifar10_fc.train(loss, global_step)
        correct_ones = cifar10_fc.correct_ones(logits=logits, labels=batch_labels)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        config = tf_utils.gpu_config(FLAGS.gpu_id)
        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            summary_writer = tf.summary.FileWriter(logdir=save_locations.summary_save_dir, graph=sess.graph)

            tf.train.start_queue_runners(sess=sess)
            for i in xrange(FLAGS.max_steps):
                _, loss_, correct_ones_ = sess.run([train_op, loss, correct_ones])

                assert not np.isnan(loss_), 'Model diverged with loss = NaN, try again'

                # if (i+1) % 10 == 0:
                print '[{:08d}|{:08d}]\tloss : {:.3f}\t, correct ones [{:d}|{:d}]'.format(i, FLAGS.max_steps,
                                                                                          loss_, correct_ones_, FLAGS.batch_size)
                if (i+1 % 100) ==0:
                    summary_ = sess.run(summary_op)
                    summary_writer.add_summary(summary_, global_step=global_step)

                if (i+1) % 500 == 0:
                    save_path= os.path.join(save_locations.model_save_dir, 'model.ckpt')
                    saver.save(sess=sess,global_step=global_step, save_path=save_path)


def main(argv=None):
    train()



if __name__ == '__main__':
    tf.app.run()