import tensorflow as tf


def main1():
    # sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    # with tf.Graph().as_default():
        sess = tf.Session()
        output=sess.run(tf.add(2,2))
        print (output)
        sess.close()

    # sess.close()

    # print tf.add(2,2).eval()
def main2():
    x = tf.placeholder(dtype=tf.float32, name='input', shape=[])
    two = tf.Variable(initial_value=2, trainable=False)
    output = tf.cast(tf.add(x,2), tf.int32)

    with tf.Session() as sess:
        output_s = output.eval(feed_dict={x: 2.4})
        print output_s

def main3():
    with tf.variable_scope('Fuck') as scope:
        x = tf.add(1,2, name=scope.name)
        print x.op.name


if __name__ == '__main__':
    main3()