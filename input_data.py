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

import os
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
# import cv2
from scipy.misc import imresize
# import time
import glob

home_dir = os.path.expanduser('~')

random.seed(0)


def get_list_of_filesnlabels(list_file):
    # home_dir = os.path.expanduser('~')
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

def get_frames_data(dirname, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []
    # s_index = 0

    abs_dirname = dirname.replace('~',home_dir)
    # print('Reading File: {:s}'.format(abs_dirname))

    fullimagenames=sorted(glob.glob(os.path.join(abs_dirname,'*.{:s}'.format('jpg'))))

    if len(fullimagenames) < num_frames_per_clip:
        print('{:s} does not have enough data'.format(abs_dirname))
        return [],[]
    #fixme: this time remove the randomness, just use the center
    # start_idx = random.randint(0, len(fullimagenames) - num_frames_per_clip)
    start_idx = int((len(fullimagenames) - num_frames_per_clip)/2)
    print('{:s} has {:d} files, starting:{:d}'.format(abs_dirname,len(fullimagenames), start_idx))

    selectedimagenames=fullimagenames[start_idx:start_idx+num_frames_per_clip]
    for single_filename in selectedimagenames:
        img = Image.open(single_filename)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, start_idx


def read_clip_and_label(filenames, labels, batch_size, np_mean, num_frames_per_clip=16, crop_size=112):
    data = []
    label = []
    for file_idx,dirname in enumerate(filenames):
        # print("Loading a video clip from {}...".format(dirname))
        tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
        img_datas = []
        if tmp_data:
            for j in xrange(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))
                if (img.width > img.height):
                    scale = float(crop_size) / float(img.height)
                    img = np.array(imresize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(
                        np.float32)
                else:
                    scale = float(crop_size) / float(img.width)
                    img = np.array(imresize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(
                        np.float32)

                img = img[int((img.shape[0] - crop_size) / 2): int((img.shape[0] - crop_size) / 2) + crop_size,
                       int((img.shape[1] - crop_size) / 2):int((img.shape[1] - crop_size) / 2) + crop_size, :] - np_mean[j]
                img_datas.append(img)
            data.append(img_datas)
            label.append(labels[file_idx])
            # label.append(int(tmp_label))
            # batch_index = batch_index + 1
            # read_dirnames.append(dirname)

    #todo: pad (duplicate) data/label if less than batch_size, here might be the reason why the low performance: data are repeated
    # valid_len = len(data)
    # pad_len = batch_size - valid_len
    # # it's not likely none of the data are satisfied
    # pad_data = data[-1]
    # pad_label = label[-1]
    # if pad_len:
    #     for i in range(pad_len):
    #         data.append(pad_data)
    #         label.append(pad_label)
            # label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)
    assert np_arr_data.shape[0] == np_arr_label.shape[0]
    # print ('valid # of frames: {:d}'.format(np_arr_data.shape[0]))
    return np_arr_data, np_arr_label
