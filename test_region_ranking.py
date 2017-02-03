# test the ranking graph
import tensorflow as tf
import numpy as np
from tf_utils import variable_with_weight_decay

def main(argv=None):

    with tf.Graph().as_default() as graph:

        np_input = np.asanyarray(range(10*16*32*32*3), dtype=np.float32)
        np_input = np.reshape(np_input, (10, 16, 32, 32, 3))
        tf_input = tf.constant(np_input)

        def region_ranking_3d(tf_input, weight_shape, strides, batch_size):
            """RegionRanking layer works similar to max-pooling

            Args:
                tf_input [batch_size, d, h, w, c]
                weight_shape [d, h, w, c, 1]  convolve with input to create weights for each component
                strids [h, w, c] shrink factor on each of the dimensions
                batch_size

            Returns:
                Output of the layer

            Rrainable variables:
                tf_ranking_w
            """
            # if not strides:
                # strides=[2, 2, 2]  # stride on depth, height, width

            # b_size, d, h, w, c = tf.shape(tf_input, tf.int32)
            shrink_factor = reduce(lambda x, y: x*y, strides)
            tf_input_shape = tf.shape(tf_input, out_type=tf.int32)

            tf_ranking_w = variable_with_weight_decay('w',
                                                      shape=weight_shape,
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      wd=None)

            tf_weights = tf.nn.conv3d(tf_input, tf_ranking_w, strides=[1, 1, 1, 1, 1], padding='SAME')

            tf_input_line = tf.reshape(tf_input, tf.pack([tf_input_shape[0], -1, tf_input_shape[-1]])) # keep the batch_size and channels
            # fixme: it has to be in batch_size dimension because we want a sort in this table
            tf_weights_line = tf.reshape(tf_weights, tf.pack([tf_input_shape[0], -1]))

            # sorted=False enables that their spatial relationships will be roughly kept
            _, tf_indices = tf.nn.top_k(tf_weights_line, k=tf.cast(tf.shape(tf_weights_line)[-1] / shrink_factor, tf.int32), sorted=False)

            tf_indices_line = tf.reshape(tf_indices, [-1])  # shape [batch_size, top_k_indices]
            tf_indices_helper = tf.expand_dims(tf.range(batch_size), 1)     # should be batch size
            tf_indices_helper = tf.tile(tf_indices_helper, multiples=tf.pack([1, tf.shape(tf_indices)[1]]))
            tf_indices_helper = tf.reshape(tf_indices_helper, [-1])

            tf_indices_2d = tf.stack([tf_indices_helper, tf_indices_line], axis=1)

            tf_input_shrinked = tf.gather_nd(tf_input_line, indices=tf_indices_2d)
            tf_output = tf.reshape(tf_input_shrinked,
                                   tf.pack([tf_input_shape[0], tf_input_shape[1]/strides[0], tf_input_shape[2] / strides[1], tf_input_shape[3] / strides[2], tf_input_shape[4]]))
            return tf_output

        tf_output = region_ranking_3d(tf_input, weight_shape=[3, 3, 3, 3, 1], strides=[2, 2, 2], batch_size=10)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, input_ = sess.run([tf_output, tf_input])
            print 'Hello world!'

# test region ranking algorithm:
# def test():
# def main(argv=None):
#
#     with tf.Graph().as_default() as graph:
#
#         tf_input = tf.constant(np.random.rand(3,4,3))
#         np_input_weight = np.asanyarray(range(3 * 4 * 1 ), dtype=np.float32)
#         np_input_weight = np.reshape(np_input_weight, (3, 4, 1))
#         tf_input_weight = tf.constant(np_input_weight)
#         tf_input_shape = tf.shape(tf_input, out_type=tf.int32)
#         tf_input_line = tf.reshape(tf_input, tf.pack([tf_input_shape[0], -1, tf_input_shape[-1]]))
#         tf_weight_line = tf.reshape(tf_input_weight, tf.pack([tf_input_shape[0], -1]))
#         _, tf_indices = tf.nn.top_k(tf_weight_line, k=tf.cast(tf.shape(tf_weight_line)[-1]/4, tf.int32),
#                                     sorted=False)
#         tf_indices_line = tf.reshape(tf_indices, [-1])
#         tf_indices_helper = tf.expand_dims(tf.range(3), 1) # should be batch size
#         tf_indices_helper = tf.tile(tf_indices_helper, multiples=tf.pack([1, tf.shape(tf_indices)[1]]))
#         tf_indices_helper = tf.reshape(tf_indices_helper, [-1])
#         tf_indices_2d = tf.stack([tf_indices_helper, tf_indices_line], axis=1)
#         tf_input_shrinked = tf.gather_nd(tf_input_line, indices=tf_indices_2d)
#         tf_output = tf.reshape(tf_input_shrinked, tf.pack(
#             [tf_input_shape[0], tf_input_shape[1] / 2, tf_input_shape[2]]))
#
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             output = sess.run(tf_output)
#             print 'Hello world!'

if __name__ == '__main__':
    tf.app.run()

