# data loader for ucf 101 dataset
import tensorflow as tf
import os
import glob

FLAGS = tf.app.flags.FLAGS

# Global constants describing the UCF 101 data set.
NEW_HEIGHT = 150
NEW_WIDTH = 200

CROP_SIZE = 128
NUM_FRAMES_PER_CLIP = 16

# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9537
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3783

IMAGE_FORMAT = 'jpg'
TF_FORMAT = 'tfrecord'


# given a directory, return batch directly fed into the graph
# todo: add evaluation part, will evaluate based on different spatial and temporal offsets
def inputs(filepath):
    filepattern = os.path.join(filepath, '*.{:s}'.format(TF_FORMAT))
    tf_filelist = tf.matching_files(filepattern)


    filenameQ = tf.train.string_input_producer(tf_filelist, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filenameQ)

    # note: corresponds to the format in c3d_prepare_tfrecords
    feature_pattern = {
        'd': tf.FixedLenFeature([], tf.int64),
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'c': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([],tf.string),
        'filename': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_single_example(
        serialized_example,features=feature_pattern)

    tf_image_d = tf.cast(features['d'], tf.int32)
    tf_image_h = tf.cast(features['h'], tf.int32)
    tf_image_w = tf.cast(features['w'], tf.int32)
    tf_image_c = tf.cast(features['c'], tf.int32)

    # image_sparse = tf.sparse_tensor_to_dense(features['image'])
    tf_image_seq = tf.decode_raw(features['image'], tf.uint8)
    tf_image_seq = tf.reshape(tf_image_seq, tf.pack([tf_image_d, tf_image_h, tf_image_w, tf_image_c]))
    tf_image_seq = tf.image.per_image_standardization(tf_image_seq)

    tf_filename = features['filename']
    tf_image_lb =features['label']

    # update 1: return the original video for evaluation, since our network is fully convo:
    tf_image_seq = tf.expand_dims(tf_image_seq, axis=0)
    return tf_image_seq, tf_image_lb, tf_filename






