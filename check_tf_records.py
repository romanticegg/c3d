# check if the tfrecord is well done:
import tensorflow as tf
import c3d_input_ucf101
import tf_utils

flags = tf.app.flags
flags.DEFINE_string('file_path', '/Users/zijwei/Dev/datasets/UCF-101-g16/train', 'Path to check the files')
flags.DEFINE_integer('batch_size', 1, 'Batch size[1]')
flags.DEFINE_integer('gpu_id', None, 'GPU [None]')

FLAGS = flags.FLAGS


def main(argv=None):
    with tf.Graph().as_default() as graph:
        # tf_images, tf_lb, tf_filename, tf_sampl_start, d, h, w, c = c3d_input_ucf101.inputs(filepath)
        tf_images, tf_lb, tf_fnames = c3d_input_ucf101.inputs(FLAGS.file_path, isTraining=True)
        config = tf_utils.gpu_config(FLAGS.gpu_id)

        with tf.Session(config=config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(10000):
                images, labels, fnames= sess.run([tf_images, tf_lb, tf_fnames])
                print 'i: {:d}, Name: {:s}, Label {:d}, image size [{:s}]'.format(i, fnames[0], labels[0], ', '.join(map(str, images.shape)))
            coord.request_stop()
            coord.join(threads=threads)



if __name__ == '__main__':
    tf.app.run()