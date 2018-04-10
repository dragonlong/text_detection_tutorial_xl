# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import utils.reader as reader
import utils.util as util
from configrnn import get_config
from modelrnn import VectModel
from input_node import DetectorInput

############ Macros ############
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

###############################
# FLAGS or args.parser
#####################################################
# Inputs with FLAGS object
# tf.flags, and flags.DEFINE_[bool, string, integer]
from tensorflow.python.client import device_lib
flags   = tf.flags
logging = tf.logging
now = datetime.now()

# the first method call in tf.flags
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, meidum, large.")
flags.DEFINE_string("data_path", "./output/", "Where data is stored" )
flags.DEFINE_string("save_path", "./model_summary/"+now.strftime("%Y%m%d-%H%M%S") + "/", "Model output")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CUDNN, "one of CUDNN: BASIC, BLOCK")

global FLAGS
FLAGS = flags.FLAGS
##############################
# Training details
##############  Main Model Running ################
###### Note how do you run this in a remote servers#######


def run_epoch(session, model, epoch, eval_op=None, verbose=False, summary_writer=None):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            #"gradients": model.grads,
            "loss": model.loss,
            "summary": model.summary_merged
            }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict=feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]
        loss = vals["loss"]
        summary = vals["summary"]
        summary_writer.add_summary(summary, step/model.input.epoch_size+epoch)
        #print(loss)

        costs += cost
        iters += model.input.num_steps

        # if verbose and step % (model.input.epoch_size // 10) == 10:
        #     print("%.3f perplexity: %.3f speed: %.0f wps" %
        #                 (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
        #                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
        #                   (time.time() - start_time)))

    costs_sum = np.sum(np.sum(costs))
    return np.exp(costs_sum / iters)


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

    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        initializer = tf.random_normal_initializer()
        with tf.name_scope("Train"):
            train_input = DetectorInput(config= config, frame_start=1, frame_end=260, FLAGS=FLAGS, name = "TrainInput")
            with tf.variable_scope("Model", reuse = None, initializer=initializer):
                m = VectModel(is_training = True, config = config, input_ = train_input)
            training_cost_sum = tf.summary.scalar("Training_Loss", m.cost)
            training_lr = tf.summary.scalar("Learning_Rate", m.lr)
            m.summary_merged = tf.summary.merge([training_lr, training_cost_sum])

        with tf.name_scope("Valid"):
            valid_input = DetectorInput(config= config, frame_start=261, frame_end = 280, FLAGS=FLAGS, name = "ValidInput")
            with tf.variable_scope("Model", reuse = True, initializer=initializer):
                mvalid = VectModel(is_training = False, config = config, input_ = valid_input)
            val_cost_sum = tf.summary.scalar("Validation_Loss", mvalid.cost)
            mvalid.summary_merged = val_cost_sum

        with tf.name_scope("Test"):# used to share variables
            test_input = DetectorInput(config= config, frame_start=281, frame_end = 301, FLAGS=FLAGS, name = "TestInput")
            with tf.variable_scope("Model", reuse = True, initializer=initializer):
                mtest = VectModel(is_training = False, config = config, input_ = test_input)
            test_cost_sum = tf.summary.scalar("Test_Loss", mtest.cost)
            mtest.summary_merged = test_cost_sum
        # Now we have got our models ready, so create a dictionary to store those computational graph
        models = {"Train":m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        # if raise ValueError()
        soft_placement = False
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)

#     with tf.Graph().as_default():
#         # import the metagraph and restore the operations
#         tf.train.import_meta_graph(metagraph)
#         for model in models.values():
#             model.import_ops(config)
        # create a global model savor or here we use the supervisor
        # train_writer = tf.summary.FileWriter(dir_summaries + '/train', sess.graph)
        # val_writer = tf.summary.FileWriter(dir_summaries + '/val')
        # summary, _ = sess.run([merged, Model.optimizer], feed_dict={x: t_img, y: t_lbl})
        # train_writer.add_summary(summary, i * num_tr + j)
        sv = tf.train.Supervisor()
        # ? maybe here it is doing something on softplacement
        config_proto = tf.ConfigProto(allow_soft_placement = soft_placement)
        with sv.managed_session(config=config_proto) as session:
            train_writer = tf.summary.FileWriter(FLAGS.save_path + 'train/', session.graph)
            val_writer = tf.summary.FileWriter(FLAGS.save_path + 'val/')
            test_writer = tf.summary.FileWriter(FLAGS.save_path + 'test/')
            # start training
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay**max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                # apply training along the way
                print("Epoch: %d Learning Rate: %.3f" % (i+1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, i, eval_op = m.train_op, summary_writer= train_writer, verbose = True)
                print("Epoch: %d training Perplexity: %.3f" % (i+1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, i, summary_writer= val_writer)
                print("Epoch: %d Valid Perplexity: %.3f" % (i+1, valid_perplexity))
                test_perplexity = run_epoch(session, mtest, i, summary_writer= test_writer)
                print("Test perplexity:%.3f" % test_perplexity)
                #
                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    #sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    tf.app.run()
