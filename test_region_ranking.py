# test the ranking graph
import tensorflow as tf
import numpy as np
from tf_utils import variable_with_weight_decay

def main(argv=None):

    with tf.Graph().as_default() as graph:

        np_input = np.asanyarray(range(10*16*32*32*3), dtype=np.float32)
        np_input = np.reshape(np_input, (10, 16, 32, 32, 3))

        tf_input = tf.constant(np_input)
        # b_size, d, h, w, c = tf.shape(tf_input, tf.int32)
        tf_input_shape = tf.shape(tf_input, out_type=tf.int32)

        k1_w = variable_with_weight_decay('w', shape=[3, 3, 3, 3, 1], initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                           wd=None)

        tf_weights = tf.nn.conv3d(tf_input, k1_w, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')

        tf_input_line = tf.reshape(tf_input, tf.pack([tf_input_shape[0], -1, tf_input_shape[4]]))
        # fixme: it has to be in batch_size dimension because we want a sort in this table
        tf_weights_line = tf.reshape(tf_weights, tf.pack([tf_input_shape[0], -1]))

        tf_weights_line_shape = tf.shape(tf_weights_line)
        tf_input_line_shape = tf.shape(tf_input_line)

        # This makes sure that their spatial relationships will be roughly kept
        _, tf_indices = tf.nn.top_k(tf_weights_line, k=tf.cast(tf_weights_line_shape[1] / 4, tf.int32), sorted=False)

        tf_indices_line = tf.reshape(tf_indices, [-1]) # shape [batch_size, top_k_indices]
        tf_indices_helper = tf.reshape(tf.range(10), [-1,1]) # should be batch size
        tf_indices_helper2 = tf.tile(tf_indices_helper, multiples=tf.pack([1, tf.shape(tf_indices)[1]]))
        tf_indices_helper3 = tf.reshape(tf_indices_helper2, [-1])

        tf_indices_2d = tf.stack([tf_indices_helper3, tf_indices_line], axis=1)

        tf_input_shrinked = tf.gather_nd(tf_input_line, indices=tf_indices_2d)
        tf_output = tf.reshape(tf_input_shrinked, tf.pack([tf_input_shape[0], tf_input_shape[1], tf_input_shape[2] / 2, tf_input_shape[3] / 2, tf_input_shape[4]]))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(tf_output)
            print 'Hello world!'

if __name__ == '__main__':
    tf.app.run()

