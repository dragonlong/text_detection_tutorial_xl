##===================== Statements & Copyright ===================##
"""
LOG:     April 8th
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project
"""
# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import collections
import os
import io
from datetime import datetime
from random import randint
from lstm.rnn_eval import model_eval
from lstm.rnn_eval import draw_illu

import lstm.utils.util as util
from lstm.configrnn import get_config
from lstm.modelrnn_new import VectModel
from lstm.input_node import DetectorInputMul

############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

###############################
# FLAGS or args.parser
#####################################################
from tensorflow.python.client import device_lib
flags   = tf.flags
logging = tf.logging
now = datetime.now()

# the first method call in tf.flags
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/"+now.strftime("%Y%m%d-%H%M%S") + "/", "Model output")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CUDNN, "one of CUDNN: BASIC, BLOCK")
flags.DEFINE_boolean("source", True, "where to get the input data")

global FLAGS
FLAGS = flags.FLAGS
##############################
# Training details
##############  Main Model Running ################
###### Note how do you run this in a remote servers#######
"""
Some notes:
The key functions for running one epoch, with multiple steps, each step a batch-size data with
num-time-steps of frames will be put into the whole model, run with [batch_size, num_steps, vect_size]
here we adopt the default size as 9*40, so we will need to resize it into 40 boxes for each one if
the confidence is not zero
Visualization will need:
1. the index of both video and current frame(to retrieve the video frame);
2. input heat-map(flatten) and reshape into the 160*160
3. predicted vector, or (targets)
"""


def run_epoch(session, model, input_,  config, epoch,  eval_op=None, verbose=False, summary_writer=None):
    """
    :param session:
    :param model:
    :param input_: with .data, .targets, .cnt_frame, .video_name
    :param config: configuration for parameters like size
    :param epoch:  current epoch number
    :param eval_op:whether or not to train and back-propagate
    :param verbose:
    :param summary_writer:which summary writter to write into
    :return:
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            "loss": model.loss,
            "summary": model.summary_merged,
            "predict": model.prediction,
            }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(config.epoch_size):
        i = randint(0, len(input_.cnt_frame)-1)
        print('choosing No. %d video, with shape [%d, %d]' % (i, input_.data[0].shape[0], input_.data[i].shape[1]))
        feed_dict = {}
        _, l1 = input_.data[i].shape
        _, l2 = input_.targets[0].shape
        data = np.zeros([config.batch_size, config.num_steps, l1])
        targets = np.zeros([config.batch_size, config.num_steps, l2])
        frame_set =[]
        if eval_op is not None:
            for k in range(config.batch_size):
                j = randint(0, input_.cnt_frame[i] - config.num_steps)
                # print('choosing starting frame %d, with num_steps is %d' % (j, config.num_steps))
                frame_set.append(j/input_.cnt_frame[i])
                data[k, :, :] = input_.data[i][j:(j+config.num_steps), :]
                targets[k, :, :] = input_.targets[i][j:(j+config.num_steps), :]
        else:
            for k in range(config.batch_size):
                j = randint(0, input_.cnt_frame[i] - config.num_steps)
                # print('choosing starting frame %d, with num_steps is %d' % (j, config.num_steps))
                frame_set.append(j / input_.cnt_frame[i])
                data[k, :, :] = input_.data[i][j:(j + config.num_steps), :]
                targets[k, :, :] = input_.targets[i][j:(j + config.num_steps), :]

        feed_dict[model.input_data] = data
        feed_dict[model.targets] = targets
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict=feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]
        loss = vals["loss"]
        predicts = vals["predict"]
        summary = vals["summary"]
        summary_writer.add_summary(summary, step/config.epoch_size+epoch)

        costs += cost
        iters += config.num_steps
        # Calculating the p, r, f1
        precision, recall, f1 = model_eval(targets, predicts, input_.video_name, i, frame_set)
        # Visualization
        if step % 20 == 0:
            plot_buf = gen_plot(input_.video_name, frame_set, data, predicts)
        #image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        # add the batch dimension
        #image = tf.expand_dims(image, 0)
        # Add image summary
        #summary_op = tf.summary.image("plot", image)
        # Run visualization
        #summary_vis = session.run(summary_op)
        #summary_writer.add_summary(summary_vis)
        if eval_op is not None:
            print('Training')
        else:
            print('Evaluation and testing')
        print('Epoch %d, step %d has precision %f , recall %f, f1 %f' %(epoch, step, precision, recall, f1))
        # if verbose and step % (model.input.epoch_size // 10) == 10:
        #     print("%.3f perplexity: %.3f speed: %.0f wps" %
        #                 (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
        #                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
        #                   (time.time() - start_time)))
        # evaluation
    costs_sum = np.sum(np.sum(costs))
    return np.exp(costs_sum / iters)


# plot the images
def gen_plot(v_name, index_frame, data, predicts):
    """Create a pyplot plot and save to buffer."""
    # currently only focusing on the very first one
    base_path = '/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process'
    video_path = base_path+'/'+v_name[0]+'.avi'
    # read the video frame and related infos
    cap = cv2.VideoCapture(video_path)
    frame_width =  int(cap.get(3))
    frame_height = int(cap.get(4))
    cap.set(int(index_frame[0]), 2)
    ret, frame = cap.read()
    # draw a new box on top of the current frame
    predict = convert_vect(predicts[0, 0, :], frame_width, frame_height)
    newimage = draw_illu(frame.copy(), predict)
    # get heatmap from data
    heatmap = np.reshape(data[0, 0, :], (160, 160))
    fig1 = plt.figure(figsize=(10, 10))
    fig1.add_subplot(1, 2, 1)
    plt.imshow(newimage)
    plt.title("Ground Truth & EAST Prediction")
    fig1.add_subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title('Input heat map')
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def convert_vect(line, w, h):
    text_lines = []
    object_num = 40
    for n in range(0, object_num):
        tl = collections.OrderedDict()
        tl['x0'] = line[n * 9]*w
        tl['y0'] = line[n * 9 + 1]*h
        tl['x1'] = line[n * 9 + 2]*w
        tl['y1'] = line[n * 9 + 3]*h
        tl['x2'] = line[n * 9 + 4]*w
        tl['y2'] = line[n * 9 + 5]*h
        tl['x3'] = line[n * 9 + 6]*w
        tl['y3'] = line[n * 9 + 7]*h
        text_lines.append(tl)
    dict_coded = {'text_lines':text_lines}
    return dict_coded


def main(_):
    # to increase the code robustness
    if not FLAGS.data_path:
        raise ValueError("Must set --")
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if (FLAGS.num_gpus > len(gpus)):
        raise ValueError("Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

    config = get_config(FLAGS)
    eval_config = get_config(FLAGS)
    eval_config.batch_size = 1
    eval_config.num_steps  = 1
    eval_config.epoch_size = 500
    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        initializer = tf.random_normal_initializer()
        with tf.name_scope("Train"):
            # use placeholder to stand for input and targets
            x_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.vocab_size])
            y_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.output_size])
            with tf.variable_scope("Model", reuse = None, initializer=initializer):
                m = VectModel(True, config, x_train, y_train)
            training_cost_sum = tf.summary.scalar("Loss", m.cost)
            training_lr = tf.summary.scalar("Learning_Rate", m.lr)
            m.summary_merged = tf.summary.merge([training_lr, training_cost_sum])
        with tf.name_scope("Val"):
            # use placeholder to stand for input and targets
            x_val = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.vocab_size])
            y_val = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.output_size])
            with tf.variable_scope("Model", reuse = True, initializer=initializer):
                mvalid = VectModel(False, eval_config, x_val, y_val)
            val_cost_sum = tf.summary.scalar("Loss", mvalid.cost)
            #val_input_image = tf.summary.image("heat_map", tf.reshape(x_val, [-1, eval_config.num_steps, 160, 160]))
            mvalid.summary_merged = tf.summary.merge([val_cost_sum])
        with tf.name_scope("Test"):
            x_test = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.vocab_size])
            y_test = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.output_size])
            with tf.variable_scope("Model", reuse = True, initializer=initializer):
                mtest = VectModel(False, eval_config, x_test, y_test)
            test_cost_sum = tf.summary.scalar("Loss", mtest.cost)
            mtest.summary_merged = test_cost_sum
        # Now we have got our models ready, so create a dictionary to store those computational graph
        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        # if raise ValueError()
        soft_placement = False
        # we could also do coding in parallel
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)
        sv = tf.train.Supervisor()
        #softplacement
        config_proto = tf.ConfigProto(allow_soft_placement = soft_placement)
        ################### load  all data into memory ###################
        if FLAGS.source is True:
            datapath = FLAGS.data_path
            train_input = DetectorInputMul(datapath,1, 8)
            val_input =  DetectorInputMul(datapath, 9, 10)
            test_input =  DetectorInputMul(datapath, 11, 12)
        else:
            pkl_train = FLAGS.data_path + '/' + 'train_input' + '.obj'
            pkl_val = FLAGS.data_path + '/' + 'val_input' + '.obj'
            pkl_test = FLAGS.data_path + '/' + 'test_input' + '.obj'
            file_train = open(pkl_train, 'r')
            train_input = pickle.load(file_train)
            file_val = open(pkl_val, 'r')
            val_input = pickle.load(file_val)
            file_test = open(pkl_test, 'r')
            test_input= pickle.load(file_test)
            print('video of train is %d , data[0] with shape [%d, %d]' % (len(train_input.video_name), train_input.data[0].shape[0], train_input.data[0].shape[1]))
        with sv.managed_session(config=config_proto) as session:
            # also save into certain folder
            train_writer = tf.summary.FileWriter(FLAGS.save_path + 'train/', session.graph)
            val_writer = tf.summary.FileWriter(FLAGS.save_path + 'val/')
            test_writer = tf.summary.FileWriter(FLAGS.save_path + 'test/')
            # start training
            for i in range(config.max_max_epoch):
                # i is the over-all epoch number
                lr_decay = config.lr_decay**max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                # apply training along the way
                print("Epoch: %d Learning Rate: %.3f" % (i+1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_input, config, i, eval_op = m.train_op, summary_writer= train_writer, verbose = True)
                print("Epoch: %d training Perplexity: %.3f" % (i+1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, val_input, eval_config, i,  summary_writer= val_writer, verbose = True)
                print("Epoch: %d Valid Perplexity: %.3f" % (i+1, valid_perplexity))
                test_perplexity = run_epoch(session, mtest, test_input, eval_config, i, summary_writer= test_writer)
                print("Test perplexity:%.3f" % test_perplexity)
                #
                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    tf.app.run()
