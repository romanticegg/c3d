import tensorflow as tf
from scipy.ndimage import imread
import PIL.Image as Image
import numpy as np
import os


from c3d_ucf101_badcase import load_pathnlabel

def test_reading_methods():
    # method 1:
    imagefile = '/Users/zijwei/Dev/datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg'
    image1 = Image.open(imagefile)
    # to the test following the reading process in get_frame_data function
    # method 2:
    frame = imread(imagefile, mode='RGB')
    print ('Conclusion: using cv2 and scipy image for image reading and so on')

def main1():
    filepath = '/Users/zijwei/Dev/datasets/UCF101-split/testlist01_01.txt'
    abs_path = '/Users/zijwei/Dev/datasets/UCF-101'
    filenames, labels = load_pathnlabel(filepath, abs_path=abs_path)
    assert  len(filenames)==len(labels)
    print '# of files {:d}'.format(len(filenames))
    print 'Done'


# for ucf-101 only
def getClassInds(filepath, abs_path=None):
    classnames = []
    classIds = []

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            line_content = line.strip('\r\n').split()
            s_classname = line_content[1]
            s_classId = int(line_content[0])

            if abs_path:
                s_classname=os.path.join(abs_path, s_classname)
            classnames.append(s_classname)
            classIds.append(s_classId)
    return classnames, classIds


# change test format to be the save the the training format
def main2():
    class_labelpath = '/Users/zijwei/Dev/datasets/UCF101-split/classInd.txt'
    filenames, labels = getClassInds(class_labelpath)
    Id_dicts= dict(zip(filenames, labels))

    # now read test files with only names:
    test_file = '/Users/zijwei/Dev/datasets/UCF101-split/testlist01.txt'
    testnames = []
    testlabels = []

    with open(test_file, 'r') as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            line_content = line.strip('\r\n').split()
            s_path = line_content[0]
            s_classname = s_path.split(os.sep)[0]
            s_label = Id_dicts[s_classname]

            testnames.append(s_path)
            testlabels.append(s_label)

    write_file = '/Users/zijwei/Dev/datasets/UCF101-split/testlist01_01.txt'
    with open(write_file, 'w') as f:
        for s_path, s_label in zip(testnames, testlabels):
            f.write('{:s} {:d}\r\n'.format(s_path, s_label))

    assert len(filenames) == len(labels)
    print '# of files {:d}'.format(len(filenames))
    print 'Done'

# test on tf records
def main3(argv=None):

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
    main1()