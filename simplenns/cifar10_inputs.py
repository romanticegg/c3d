import tensorflow as tf
import os


# flags =tf.app.flags
# flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/cifar10-batch', 'directory to save training data[/Users/zijwei/Dev/datasets]')


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

#todo: write this file for loading single file from a filename... should be the most important and only code block that you need to rewrite for a new dataset
def decode_single_image(filename_queue):
    # declare a container like stuff...
    class SingleRecord(object):
        pass

    result = SingleRecord()

    label_bytes = 1
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    # fixme: here is highly dataset specific, should be avoided
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

# given a directory, return batch directly fed into the graph
def inputs(data_dir, batch_size, isTraining=True, isRandom=False):
    if isTraining:
        filenames = [os.path.join(data_dir, 'data_batch_{:d}.bin'.format(i)) for i in xrange(1,6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = os.path.join(data_dir, 'test_batch.bin')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)


    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = decode_single_image(filename_queue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    resized_image  = tf.image.resize_image_with_crop_or_pad(reshaped_image, IMAGE_SIZE, IMAGE_SIZE)

    if isRandom:
        # Randomly flip the image horizontally.
        resized_image = tf.image.random_flip_left_right(resized_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        resized_image = tf.image.random_brightness(resized_image,
                                                     max_delta=63)
        resized_image = tf.image.random_contrast(resized_image,
                                                   lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(resized_image)

    # min_fraction_of_examples_in_queue = 0.1
    # min_queue_examples = int(num_examples_per_epoch *
    #                          min_fraction_of_examples_in_queue)

    min_queue_examples = 32
    num_preprocess_thread = 16

    if isTraining:
        batch_images, batch_labels = tf.train.shuffle_batch([float_image, read_input.label], batch_size= batch_size,
                                                            num_threads= num_preprocess_thread, enqueue_many=False, capacity=min_queue_examples+3*batch_size,
                                                            min_after_dequeue=min_queue_examples)
    else:
        batch_images, batch_labels = tf.train.batch([float_image, read_input.label], batch_size= batch_size,
                                                            num_threads= num_preprocess_thread, enqueue_many=False, capacity=min_queue_examples+3*batch_size)

    batch_labels = tf.reshape(batch_labels, [batch_size])

    return batch_images, batch_labels