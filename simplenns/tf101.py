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

# test on tensorflow copy
def main4():
    a = tf.get_variable(name='start', shape=[],initializer=tf.constant_initializer(0))
    b = a
    a = a+tf.constant(2.0)
    with tf.Session()as sess:
        sess.run(tf.initialize_variables(tf.all_variables()))
        b_val, a_val = sess.run([b,a])
        print 'b : {:f}\t, a : {:f}'.format(b_val, a_val)

if __name__ == '__main__':
    main4()