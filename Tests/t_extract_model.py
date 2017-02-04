# test on extracting tensorflow old models from a file
import tensorflow as tf
import tensorflow.python.tools as tpt


def main(argv=None):
    with tf.Graph().as_default() as graph:
        # saver = tf.train.Saver()
        model_path = '/Users/zijwei/Dev/C3D-tensorflow/models/c3d_ucf_model-99.meta' # this load the graph
        saver = tf.train.import_meta_graph(model_path)
        print 'Saver Loaded'
        with tf.Session() as sess:
            saver.restore(sess,'/Users/zijwei/Dev/C3D-tensorflow/models/c3d_ucf_model-99')
            # obj = saver.restore(sess, model_path)
            print 'Hello world'

if __name__ == '__main__':
    tf.app.run()