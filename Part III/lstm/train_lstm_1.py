# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime
from random import randint
from lstm.rnn_eval import model_eval

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
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, meidum, large.")
flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/"+now.strftime("%Y%m%d-%H%M%S") + "/", "Model output")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CUDNN, "one of CUDNN: BASIC, BLOCK")
flags.DEFINE_boolean("source", True, "where to get the input data" )

global FLAGS
FLAGS = flags.FLAGS


##############################
# Training details
##############  Main Model Running ################
###### Note how do you run this in a remote servers#######
def run_epoch(session, model, input_,  config, epoch,  eval_op=None, verbose=False, summary_writer=None):
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
        precision, recall, f1 = model_eval(targets, predicts, input_.video_name, i, frame_set)
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
            val_input_image = tf.summary.image("heat_map", tf.reshape(x_val, [-1, eval_config.num_steps, 160, 160]))
            mvalid.summary_merged = tf.summary.merge([val_cost_sum, val_input_image])
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
                valid_perplexity = run_epoch(session, mvalid, val_input, eval_config,i,  summary_writer= val_writer)
                print("Epoch: %d Valid Perplexity: %.3f" % (i+1, valid_perplexity))
                test_perplexity = run_epoch(session, mtest, test_input, eval_config, i, summary_writer= test_writer)
                print("Test perplexity:%.3f" % test_perplexity)
                #
                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    tf.app.run()
