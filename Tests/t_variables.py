import tensorflow as tf


def main(argv=None):

    with tf.Graph().as_default() as graph:
        v1 = tf.Variable(1)
        var = tf.trainable_variables()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            v1_, var_ = sess.run([v1, var])
            print 'Hello world'


if __name__ == '__main__':
    tf.app.run()