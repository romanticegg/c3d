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
def inputs(filepath, isTraining=True):
    filepattern = os.path.join(filepath, '*.{:s}'.format(TF_FORMAT))
    tf_filelist = tf.matching_files(filepattern)

    if isTraining:
        filenameQ = tf.train.string_input_producer(tf_filelist, shuffle=True)
    else:
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

    tf_filename = features['filename']
    tf_image_lb =features['label']

    #todo: processing images when evaluating
    # get length to be 16

    sample_start =tf.cond(tf_image_d > NUM_FRAMES_PER_CLIP,
                          lambda: tf.random_uniform([], minval=0, maxval=tf_image_d-NUM_FRAMES_PER_CLIP, dtype=tf.int32),
                          lambda: tf.constant(0))
    tf_image_seq = tf.slice(tf_image_seq, [sample_start, 0, 0, 0],
                            [NUM_FRAMES_PER_CLIP, tf_image_h, tf_image_w, tf_image_c])
    # resize
    tf_image_seq = tf.image.resize_images(tf_image_seq, [NEW_HEIGHT, NEW_WIDTH])

    # current: [d, h, w, c]  --> [h, w, c, d]
    tf_image_seq = tf.transpose(tf_image_seq, [1, 2, 3, 0])
    # [h, w, c, d] --> [h, w, c*d]
    tf_image_seq= tf.reshape(tf_image_seq, [NEW_HEIGHT, NEW_WIDTH, NUM_FRAMES_PER_CLIP*3])
    # update 1: only normal cropping:
    # tf_image_seq = tf.image.resize_image_with_crop_or_pad(tf_image_seq, CROP_SIZE, CROP_SIZE)

    # update 2: random cropping: randomly crop a small region and resize to CROP_SIZE
    # here NEW_HEIGHT is larger than
    # https://arxiv.org/pdf/1507.02159v1.pdf
    # https://arxiv.org/pdf/1604.04494v1.pdf
    RANDOM_CROP_RATIO = tf.constant([1.0, 0.875, 0.75, 2.0 / 3], dtype=tf.float32)

    tf_w_size = tf.cast(NEW_HEIGHT * RANDOM_CROP_RATIO[tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)], tf.int32)

    tf_h_size = tf.cast(NEW_HEIGHT * RANDOM_CROP_RATIO[tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)], tf.int32)

    tf_image_seq = tf.image.resize_image_with_crop_or_pad(tf_image_seq, tf_h_size, tf_w_size)

    tf_image_seq = tf.image.resize_images(tf_image_seq, [CROP_SIZE, CROP_SIZE])

    # randome image operation:
    tf_image_seq = tf.image.random_flip_left_right(tf_image_seq)
    # fixme: should we also subtract mean?
    tf_image_seq = tf.image.per_image_standardization(tf_image_seq)

    # move back to [d, h, w, c]
    tf_image_seq = tf.reshape(tf_image_seq, [CROP_SIZE, CROP_SIZE, 3, NUM_FRAMES_PER_CLIP])
    tf_image_seq = tf.transpose(tf_image_seq, [3, 0, 1, 2])

    min_queue_examples = 32
    num_preprocess_thread = 16

    if isTraining:
        batch_images, batch_labels, batch_filenames = tf.train.shuffle_batch([tf_image_seq, tf_image_lb, tf_filename], batch_size=FLAGS.batch_size,
                                                            num_threads=num_preprocess_thread, enqueue_many=False,
                                                            capacity=min_queue_examples + 3 * 10,
                                                            min_after_dequeue=min_queue_examples, allow_smaller_final_batch=False)
    else:
        batch_images, batch_labels, batch_filenames = tf.train.batch([tf_image_seq, tf_image_lb, tf_filename], batch_size=FLAGS.batch_size,
                                                    num_threads=num_preprocess_thread, enqueue_many=False,
                                                    capacity=min_queue_examples + 3 * FLAGS.batch_size, allow_smaller_final_batch=False)

    return batch_images, batch_labels, batch_filenames


