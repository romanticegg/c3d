# save data to tfrecord formats
import tensorflow as tf
import glob
import os
from scipy.ndimage import imread
import numpy as np
import utils


def load_pathnlabel(filepath, abs_path=None, strip_symb=None):

    if not strip_symb:
        strip_symb = '\r\n'
    if not abs_path:
        abs_path = ''

    filenames = []
    labels = []

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            line_content = line.strip(strip_symb).split()
            s_path, ext = os.path.splitext(line_content[0])     # keep out of the .avi

            s_label = int(line_content[1])-1        #id starting from 0

            if abs_path:
                s_path=os.path.join(abs_path, s_path)
            filenames.append(s_path)
            labels.append(s_label)
    return filenames, labels


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

flags = tf.app.flags
flags.DEFINE_string('annofile_path', '/Users/zijwei/Dev/datasets/UCF101-split/testlist01_01.txt',
                    'annotation file in format (filename label)[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("abs_path", '/Users/zijwei/Dev/datasets/UCF-101', "added to filename in annotation file to make a valid path")
flags.DEFINE_string('save_dir', '/Users/zijwei/Dev/datasets/UCF-101-g16/test', 'save to [/Users/zijwei/Dev/datasets/UCF-101-g16/test]')
flags.DEFINE_string('image_format', 'jpg', 'frame format in directories[jpg]')
flags.DEFINE_string('tf_format', 'tfrecord', 'saved tfrecord format[tfrecord]')
flags.DEFINE_integer('min_len', 16, 'min length of videos [16]')

FLAGS = flags.FLAGS


def main(argv=None):

    save_dir = utils.get_dir(FLAGS.save_dir)

    file_paths, labels = load_pathnlabel(FLAGS.annofile_path, abs_path=FLAGS.abs_path)
    assert len(file_paths) == len(labels)

    wrtie_videoframes_to_tfrecord(file_paths, labels, save_dir, frame_limit=FLAGS.min_len)


def wrtie_videoframes_to_tfrecord(file_paths, labels, save_dir, frame_limit=None):

    assert len(file_paths) == len(labels), 'Files and labels mismatch'
    print 'Total # of files {:d}'.format(len(file_paths))

    save_dir = utils.get_dir(save_dir)

    # filter away
    if frame_limit:
        filtered_pairs = [(f, l) for f, l in zip(file_paths, labels)
                          if len(glob.glob(os.path.join(f,'*.{:s}'.format(FLAGS.image_format)))) >= frame_limit]

        file_paths, labels = zip(*filtered_pairs)
        assert len(file_paths) == len(labels), 'Checking filtering'

    n_files = len(file_paths)
    print '# of valid videos: {:d}'.format(n_files)

    for i, s_label, s_filename in zip(range(n_files), labels, file_paths):

        filename_stem = s_filename.split(os.sep)[-1]
        print 'Processing {:d} | {:d} \t {:s}'.format(i, n_files, filename_stem)
        file_list = glob.glob(os.path.join(s_filename, '*.{:s}'.format(FLAGS.image_format)))
        n_images = len(file_list)
        images = []

        tf_save_name = os.path.join(save_dir, '{:s}.{:s}'.format(filename_stem, FLAGS.tf_format))
        writer = tf.python_io.TFRecordWriter(tf_save_name)

        for single_filename in file_list:
            img = imread(single_filename, mode='RGB')
            images.append(img)

        np_images = np.array(images).astype(np.uint8)
        seq_shape = np_images.shape
        seq_d = seq_shape[0]    # depth, length of seq
        assert n_images==seq_d
        seq_h = seq_shape[1]    # height
        seq_w = seq_shape[2]    # width
        seq_c = seq_shape[3]    # channels
        image_raw = np_images.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'd': _int64_feature(seq_d),
            'h': _int64_feature(seq_h),
            'w': _int64_feature(seq_w),
            'c': _int64_feature(seq_c),
            'label': _int64_feature(int(s_label)),
            'image': _bytes_feature(image_raw),
            'filename': _bytes_feature(filename_stem)}))

        writer.write(example.SerializeToString())

    print 'Done'




if __name__ == "__main__":
    tf.app.run()