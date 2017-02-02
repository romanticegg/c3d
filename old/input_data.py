# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random

import cv2
import numpy as np
from scipy.ndimage  import imread

from old import consts as c

home_dir = os.path.expanduser('~')

random.seed(0)


def get_list_of_filesnlabels(list_file):
    '''

    :param list_file: path to the file with the following format:
     path1 label1\n
     path2 label2\n
    :return: idx: 0~#, file_names: list, lables: list
    '''

    file_names = []
    labels = []

    with open(list_file, 'r') as f:
        for single_line in f:
            line_content = single_line.strip('\n').split()
            abs_dirname = line_content[0].replace('~', home_dir)
            label = int(line_content[1])
            file_names.append(abs_dirname)
            labels.append(label)

    assert len(file_names) == len(labels)
    idxs = range(len(file_names))
    return idxs, file_names, labels


def read_clip_and_label(dirnames, labels, np_mean, num_frames_per_clip=16, crop_size=112, RGB=True):
    batch_data = []
    batch_label = []

    for s_label, s_dirname in zip(labels,dirnames):
        img_seq = []
        raw_data= get_frames_data(s_dirname, num_frames_per_clip, isRGB=RGB)
        if raw_data:
            assert len(raw_data) == num_frames_per_clip
            for j in xrange(num_frames_per_clip):
                img = np.array(raw_data[j],np.float32)
                img_width = img.shape[1]
                img_height = img.shape[0]
                if img_width > img_height:
                    scale = float(crop_size) / float(img_height)
                    img = cv2.resize(np.array(img), (int(img_width * scale + 1), crop_size))
                else:
                    scale = float(crop_size) / float(img_width)
                    img = cv2.resize(np.array(img), (crop_size, int(img_height * scale + 1)))

                img = img[int((img.shape[0] - crop_size) / 2): int((img.shape[0] - crop_size) / 2) + crop_size,
                       int((img.shape[1] - crop_size) / 2):int((img.shape[1] - crop_size) / 2) + crop_size, :] - np_mean[j]

                img_seq.append(img)
            batch_data.append(img_seq)
            batch_label.append(s_label)

    np_arr_data = np.array(batch_data).astype(np.float32)
    np_arr_label = np.array(batch_label).astype(np.int64)
    assert np_arr_data.shape[0] == np_arr_label.shape[0]
    return np_arr_data, np_arr_label


def get_frames_data(abs_dirname, num_frames_per_clip=16, isRGB=True):
    '''
    Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays
    :param abs_dirname:
    :param num_frames_per_clip:
    :param isRGB:
    :return:
    '''

    fullimagenames=sorted(glob.glob(os.path.join(abs_dirname,'*.{:s}'.format(c.IMAGE_FMT))))

    if len(fullimagenames) < num_frames_per_clip:
        print('{:s} has {:d} frames,  NOT enough data'.format(abs_dirname, len(fullimagenames)))
        return []

    ret_arr = []
    #todo: currently it removes the randomness, just use the center
    # start_idx = random.randint(0, len(fullimagenames) - num_frames_per_clip)
    start_idx = int((len(fullimagenames) - num_frames_per_clip)/2)
    selectedimagenames=fullimagenames[start_idx:start_idx+num_frames_per_clip]
    for single_filename in selectedimagenames:
        img = imread(single_filename, mode='RGB')
        if not isRGB:
            img = img[:,:,::-1]
        ret_arr.append(img)

    print('{:s} has {:d} files, starting at:{:d}'.format(abs_dirname,len(fullimagenames), start_idx))

    return ret_arr
