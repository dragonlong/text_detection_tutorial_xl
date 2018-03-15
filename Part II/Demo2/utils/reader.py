  # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import xmltodict
import numpy as np
import json
import pandas as pd
from skimage import measure
from pandas.io.json import json_normalize #package for flattening json in pandas df
import tensorflow as tf

Py3 = sys.version_info[0] == 3


def read_XML(filepath, f_start, f_end, cap=40, f_w=720, f_h=480):
    with open(filepath+'XML/Video_16_3_2_GT.xml') as fd:
        doc = xmltodict.parse(fd.read())
    #print(doc['Frames']['frame'])
    #print(doc['Frames']['frame'][0])
    #print(doc['Frames']['frame'][0]['object'][0]['Point'][0]['@x'])
    #doc['Frames']['frame'][index]['@ID']
    #frame_num = len(doc['Frames']['frame'])
    frame_num = f_end- f_start+1
    target = np.zeros((frame_num, 9*cap))
    for i in range(f_start, f_end+1):
        print('%d th frame', i)
        object_num = len(doc['Frames']['frame'][i-1]['object'])
        for j in range(0, object_num):
            print(j)
            target[i - f_start, j*9]   = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x'])/f_w
            target[i - f_start, j*9+1] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y'])/f_h
            target[i - f_start, j*9+2] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x'])/f_w
            target[i - f_start, j*9+3] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y'])/f_h
            target[i - f_start, j*9+4] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x'])/f_w
            target[i - f_start, j*9+5] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y'])/f_h
            target[i - f_start, j*9+6] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x'])/f_w
            target[i - f_start, j*9+7] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y'])/f_h
            target[i - f_start, j*9+8] = 1.0
    return target


def read_json_npy(filepath, index):
    with open(filepath+'json/frame'+format(index, '03d')+'.json') as f:
        d = json.load(f)
    data = np.load(filepath+'npy/frame'+format(index, '03d')+'.npy')
    return d, data


def fusion_data(d, data_shrink, cap=15, f_w=720, f_h=480):
    d1 = np.ndarray.flatten(data_shrink)
    # with default 40 at most
    dim = cap*9
    output = np.zeros((len(d1)+dim),dtype=np.float32)
    for i in range(0, len(d['text_lines'])):
        output[i*9] = d['text_lines'][i]['x0']/f_w
        output[i*9+1] = d['text_lines'][i]['y0']/f_h
        output[i*9+2] = d['text_lines'][i]['x1']/f_w
        output[i*9+3] = d['text_lines'][i]['y1']/f_h
        output[i*9+4] = d['text_lines'][i]['x2']/f_w
        output[i*9+5] = d['text_lines'][i]['y2']/f_h
        output[i*9+6] = d['text_lines'][i]['x3']/f_w
        output[i*9+7] = d['text_lines'][i]['y3']/f_h
        output[i*9+8] = d['text_lines'][i]['score']
    output[dim:] = d1
    return output


def vect_producer(datapath, frame_start, frame_end, batch_size, num_steps, name=None):
    d, data = read_json_npy(datapath, frame_start)
    data_shrink = measure.block_reduce(np.squeeze(data), (10, 10, 1), np.mean)
    vect_encoded = fusion_data(d, data_shrink)
    l1 = len(vect_encoded)
    frame_num = frame_end - frame_start+1
    vect_set = np.zeros((frame_num, l1))
    vect_set[0, :] = vect_encoded
    for index in range(frame_start+1, frame_end+1):
        d, data = read_json_npy(datapath, index)
        data_shrink  = measure.block_reduce(np.squeeze(data), (10, 10, 1), np.mean)
        vect_encoded = fusion_data(d, data_shrink)
        vect_set[index-frame_start, :] = vect_encoded
    target = read_XML(datapath, frame_start, frame_end)
    _, l2 = target.shape
    print(vect_set.shape)
    print(target.shape)
    # convert to tensor and using shuffle queue
    with tf.name_scope(name, 'VectProducer', [batch_size, num_steps]):
        input_t = tf.convert_to_tensor(vect_set, name="input", dtype=tf.float32)
        target_t = tf.convert_to_tensor(target, name="target", dtype=tf.float32)
        batch_len  = frame_num // batch_size
        epoch_size = (batch_len - 1) // num_steps
        # parsing data from [frame_num, vect_length] to [batch_size, num_steps, vocal_size ]
        data_in  = tf.reshape(input_t, [batch_size, batch_len, l1])
        data_gt  = tf.reshape(target_t, [batch_size, batch_len, l2])
        # produce an iterable object for index
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data_in, [0, i * num_steps, 0], [batch_size, (i + 1) * num_steps, l1])
        x.set_shape([batch_size, num_steps, l1])
        y = tf.strided_slice(data_gt, [0, i * num_steps, 0], [batch_size, (i + 1) * num_steps, l2])
        y.set_shape([batch_size, num_steps, l2])
        return x, y
        # reshape


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
        data_path: string path to the directory where simple-examples.tgz has
            been extracted.

    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps]. The second element
        of the tuple is the same data time-shifted to the right by one.

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                                            [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                                                  [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                                                  [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
