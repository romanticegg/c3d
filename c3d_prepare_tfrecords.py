import tensorflow as tf
import c3d_input_ucf101
from c3d_ucf101_badcase import  load_pathnlabel
import glob
import os
from scipy.ndimage  import imread
import numpy as np
import utils

NUM_FRAMES_PER_CLIP = 16
IMAGE_FORMAT = 'jpg'
TF_RECORD_DIR = '/Users/zijwei/Dev/datasets/UCF-101-16-tfrecords'
TF_FORMAT ='tfrecord'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main1():
    filepath = '/Users/zijwei/Dev/datasets/UCF101-split/testlist01_01.txt'
    abs_path = '/Users/zijwei/Dev/datasets/UCF-101'

    save_dir = utils.get_dir(TF_RECORD_DIR)

    filenames, labels = load_pathnlabel(filepath, abs_path=abs_path)
    assert  len(filenames)==len(labels)

    filenames_patrn = []
    updated_labels = []
    for s_label, s_filename in zip(labels, filenames):
        if len(glob.glob(os.path.join(s_filename,'*.{:s}'.format(IMAGE_FORMAT))))<NUM_FRAMES_PER_CLIP:
            # print '{:s} do NOT have more than {:d} frames'.format(s_filename, NUM_FRAMES_PER_CLIP)
            continue
        filenames_patrn.append(os.path.join(s_filename, '*.{:s}'.format(IMAGE_FORMAT)))
        updated_labels.append(s_label)
    assert len(updated_labels)==len(filenames_patrn)

    print '# of files {:d}, valid ones: {:d}'.format(len(filenames), len(filenames_patrn))

    for i, s_label, s_filename_pattern in zip(range(len(filenames_patrn)), updated_labels, filenames_patrn):
        filename_stem = s_filename_pattern.split(os.sep)[-2]

        print 'Processing {:d} | {:d} \t {:s}'.format(i, len(filenames_patrn), filename_stem)
        file_list = glob.glob(s_filename_pattern)
        n_images = len(file_list)
        image_seq = []

        tf_save_name = os.path.join(save_dir, '{:s}.{:s}'.format(filename_stem, TF_FORMAT))
        writer = tf.python_io.TFRecordWriter(tf_save_name)
        for single_filename in file_list:
            img = imread(single_filename, mode='RGB')
            image_seq.append(img)

        np_image_seq = np.array(image_seq).astype(np.uint8)
        seq_shape = np_image_seq.shape
        seq_d = seq_shape[0]    # depth, length of seq
        seq_h = seq_shape[1]    # height
        seq_w = seq_shape[2]    # width
        seq_c = seq_shape[3]    # channels
        image_raw = np_image_seq.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'd': _int64_feature(seq_d),
            'h': _int64_feature(seq_h),
            'w': _int64_feature(seq_w),
            'c': _int64_feature(seq_c),
            'label': _int64_feature(int(s_label)),
            'image': _bytes_feature(image_raw),
            'filename': _bytes_feature(filename_stem)}))

        writer.write(example.SerializeToString())

        print 'Debug'

    print 'Done'


def main(argv=None):

    with tf.Graph().as_default() as graph:
        filepath = '/Users/zijwei/Dev/datasets/UCF-101-16-tfrecords'
        # tf_images, tf_lb, tf_filename, tf_sampl_start, d, h, w, c = c3d_input_ucf101.inputs(filepath)
        tf_images, tf_lb= c3d_input_ucf101.inputs(filepath)
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(100):
                images, labels= sess.run([tf_images, tf_lb])
                # print 'i: {:d}, Name: {:s}, Label {:d}, original len: {:d}, sample_start: {:d}, image size [{:s}]'.format(i, filename, labels, d_, sample_start, ', '.join(map(str, images.shape)))
                print 'Debug'
            coord.request_stop()
            coord.join(threads=threads)

if __name__ == "__main__":
    tf.app.run()