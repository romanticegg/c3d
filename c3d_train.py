import tensorflow as tf
import c3d_model_simple as c3d_model
import c3d_ucf101_badcase

# test model is successfully created
def main1(argv=None):
    with tf.Graph().as_default() as graph:

        inputs = tf.random_normal(shape=[10, c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, c3d_model.CHANNELS])
        logits = c3d_model.inference_c3d(inputs=inputs, isTraining=True)

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            logits_ = sess.run(logits)
            print 'Size of output is [{:s}]'.format(', '.join(map(str, logits_.shape)))



def main(argv=None):

    filepath = '/Users/zijwei/Dev/datasets/UCF101-split/testlist01_01.txt'
    abs_path = '/Users/zijwei/Dev/datasets/UCF-101'

    with tf.Graph().as_default() as graph:

        [Qinit, s_filename, s_label] = c3d_ucf101_badcase.inputs(filepath=filepath, abs_path=abs_path)
        listofImages, s_fileQ = c3d_ucf101_badcase.tf_loadVideoFromFile(s_filename)
        # n_files = tf.shape(matchingfiles)[0]
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            sess.run(Qinit)
            for i in range(200):
                # qr =tf.train.QueueRunner(s_fileQ)
                # coord = tf.train.Coordinator()
                # enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
                # sess.run(s_fileQ)
                s_filename_ ,image_list_ = sess.run([s_filename, listofImages])

                print '{:d} Name {:s}, size: [{:s}]'.format(i, s_filename_, ', '.join(map(str, image_list_.shape)))
                # coord.request_stop()
                # coord.join(enqueue_threads)
            # for i in range(0):
                # [filename_, label_] = sess.run([s_filename, s_label])
            # print '{:s} \t {:d}'.format(filename_, label_)


def main3(argv=None):
    with tf.Graph().as_default() as graph:

        i0 = tf.constant(0)
        m0 = tf.ones([2, 2])
        m1 = tf.ones([2, 2])
        i_limit = tf.constant(10)
        c = lambda i, m: i < i_limit
        b = lambda i, m: [i + 1, tf.concat(0, [m, m1])]
        r = tf.while_loop(
            c, b, loop_vars=[i0, m0],
             shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])])

        with tf.Session() as sess:
            output = sess.run(r)
            print 'Debug'
if __name__ == '__main__':
    tf.app.run()