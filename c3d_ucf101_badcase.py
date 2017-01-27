# data loader for ucf 101 dataset
# todo: Not done yet
import tensorflow as tf
import os
import glob

FLAGS =tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
CROP_SIZE = 128
NUM_CLASSES = 101
NUM_FRAMES_PER_CLIP = 16

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9537
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3783

IMAGE_FORMAT = 'jpg'
#filepath = '/Users/zijwei/Dev/datasets/UCF101-split/trainlist01.txt'
def load_pathnlabel(filepath, abs_path=None):
    filenames = []
    labels = []

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            line_content = line.strip('\r\n').split()
            s_path, ext = os.path.splitext(line_content[0]) # keep out of the .avi

            s_label = int(line_content[1])-1  #id starting from 0

            if abs_path:
                s_path=os.path.join(abs_path, s_path)
            filenames.append(s_path)
            labels.append(s_label)
    return filenames, labels



# #todo: write this file for loading single file from a filename... should be the most important and only code block that you need to rewrite for a new dataset
# def decode_single_image(filename_queue):
#     # declare a container like stuff...
#     class SingleRecord(object):
#         pass
#
#     result = SingleRecord()
#
#     label_bytes = 1
#     result.height = IMAGE_SIZE
#     result.width = IMAGE_SIZE
#     result.depth = 3
#     image_bytes = result.height * result.width * result.depth
#     record_bytes = label_bytes + image_bytes
#
#     reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#     result.key, value = reader.read(filename_queue)
#
#     record_bytes = tf.decode_raw(value, tf.uint8)
#     # fixme: here is highly dataset specific, should be avoided
#     result.label = tf.cast(
#         tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
#     # The remaining bytes after the label represent the image, which we reshape
#     # from [depth * height * width] to [depth, height, width].
#     depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
#                              [result.depth, result.height, result.width])
#     # Convert from [depth, height, width] to [height, width, depth].
#     result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#     return result


# given a directory, return batch directly fed into the graph
def inputs(filepath, abs_path, isTraining=True, isRandom=False):

    [filenames, labels] = load_pathnlabel(filepath, abs_path=abs_path)


    # update based on existence:
    updated_filenames = []
    updated_labels = []
    for i, s_filename in enumerate(filenames):
        if len(glob.glob(os.path.join(s_filename,'*.{:s}'.format(IMAGE_FORMAT))))<NUM_FRAMES_PER_CLIP:
            # print '{:s} do NOT have more than {:d} frames'.format(s_filename, NUM_FRAMES_PER_CLIP)
            continue
        updated_filenames.append(os.path.join(s_filename, '*.{:s}'.format(IMAGE_FORMAT)))
        updated_labels.append(labels[i])

    assert len(updated_labels)==len(updated_filenames)

    tf_filenames = tf.convert_to_tensor(updated_filenames, dtype=tf.string)
    tf_labels = tf.convert_to_tensor(updated_labels, dtype=tf.int32)

    print 'Found {:d} files with acceptable length'.format(len(updated_filenames))

    # file_bundles = zip(updated_filenames, updated_labels)
    # Create a queue that produces the filenames to read.
    idx_list = (range(len(updated_filenames)))

    idQ = tf.FIFOQueue(len(updated_filenames), tf.int32)

    Qinit = idQ.enqueue_many((idx_list,))

    s_id = idQ.dequeue()
    s_filename = tf_filenames[tf.cast(s_id, tf.int32)]
    s_label = tf_labels[tf.cast(s_id, tf.int32)]

    return Qinit, s_filename, s_label

def tf_loadVideoFromFile(tf_dirname):
    # tf_pattern = tf.concat([tf_dirname, '*.jpg'], 0)
    tf_matchingfiles = tf.matching_files(tf_dirname)

    n_files = tf.shape(tf_matchingfiles)[0]

    fileQ = tf.FIFOQueue(100, tf.string)
    fQinit = fileQ.enqueue_many((tf_matchingfiles,))
    with tf.control_dependencies([fQinit]):
        # s_file = fileQ.dequeue()


        reader = tf.WholeFileReader()
        _, value = reader.read(fileQ)

        image_seq = tf.image.decode_jpeg(value,channels=3)
        image_seq = tf.expand_dims(image_seq, axis=0)
    # image_seq = tf.ones([2,2,2])
        i = tf.constant(1)

        def condition(i, m):
            return tf.less(i, n_files)

        def body(i, m):
            # s_file =fileQ.dequeue()
            _, value = reader.read(fileQ)
            tf_s_image = tf.image.decode_jpeg(value, channels=3)
            tf_s_image = tf.expand_dims(tf_s_image, axis=0)

            # tf_s_image = tf.ones([2, 2, 2])
            return i+1, tf.concat(concat_dim=0, values=[m, tf_s_image])

    _, loaded_images = tf.while_loop(cond=condition, body=body, loop_vars=[i, image_seq],
                                  shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None, None])])

    return loaded_images, fileQ
