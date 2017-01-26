import tensorflow as tf
import c3d_model_simple as c3d_model


def main(argv=None):
    with tf.Graph().as_default() as graph:

        inputs = tf.random_normal(shape=[10, c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, c3d_model.CHANNELS])
        logits = c3d_model.inference_c3d(inputs=inputs, isTraining=True)

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            logits_ = sess.run(logits)
            print 'Size of output is [{:s}]'.format(', '.join(map(str, logits_.shape)))




if __name__ == '__main__':
    tf.app.run()