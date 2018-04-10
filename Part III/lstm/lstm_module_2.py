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
import lstm.utils.reader as reader
import lstm.utils.util as util

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
flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, meidum, large.")
flags.DEFINE_string("data_path", "./output/", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/"+now.strftime("%Y%m%d-%H%M%S") + "/", "Model output")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CUDNN, "one of CUDNN: BASIC, BLOCK")

FLAGS = flags.FLAGS
###############################
# Config file
####################################################
def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        pass
    elif FLAGS.model == "large":
        pass
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
        config.rnn_mode = BASIC
    return config


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 0.01
    max_grad_norm = 1
    num_layers    = 1
    num_steps     = 3
    output_size = 360
    hidden_size   = 200
    max_max_epoch = 500
    max_epoch     = 100
    keep_prob     = 1.0
    lr_decay      = 0.5
    batch_size    = 1
    vocab_size    = 6663
    rnn_mode      = BLOCK


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 2
    num_steps = 3
    output_size = 360
    hidden_size = 1024
    max_epoch = 100
    max_max_epoch = 500
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 6663
    rnn_mode = CUDNN
###############################
# Input function to return input_data, targets
###################################################


class DetectorInput(object):
    """The input data."""
    def __init__(self, config, frame_start, frame_end, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (((frame_end- frame_start) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.vect_producer(FLAGS.data_path, frame_start, frame_end, batch_size, num_steps, name=name)


###################################################
# Model definition for video processing
###################################################
class VectModel(object):
    """Model used for PTB processing"""
    # here input is totally an object with all kinds of features created by Input class,
    # which use reader functions
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        output_size = config.output_size
        # inputs dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(input_, config.keep_prob)
        # build up the model itself with lower-level function
        output, state = self._build_rnn_graph(input_.input_data, config, is_training)

        softmax_w = tf.get_variable(
                "softmax_w", [size, output_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [output_size], dtype=tf.float32)
        pred      = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        pred      = tf.reshape(pred, [self.batch_size, self.num_steps, output_size])
       # Use the contrib sequence loss and average over the batches
       #  loss = tf.contrib.seq2seq.sequence_loss(
       #          logits,
       #          input_.targets,
       #          tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
       #          average_across_timesteps=False,
       #          average_across_batch=True)

        loss = tf.square(pred - input_.targets)
        self._loss = loss
        self._cost = tf.reduce_mean(loss)
        self._final_state = state
        if not is_training:
            return

        # training details
        # since _lr is a variable, so we could assign number to it later by assignment
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),config.max_grad_norm)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # how to manipulate the training gradient, the optimizer actually gives us an function to do that
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())
        # so model also includes these options to get access to our training parameters, which mainly comes from
        # config function
        #self._grads = grads
        self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell"""
        # here we want to pemute the dimensions
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units =config.hidden_size,
            input_size=config.vocab_size,
            dropout=1 - config.keep_prob if is_training else 0
        )
        # what is this used for
        #params_size_t = self.
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
                "lstm_params",
                initializer=tf.random_uniform(
                        [params_size_t], -config.init_scale, config.init_scale),
                validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                                  tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                                  tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                               reuse = not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias = 0.0)
        raise valueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonial LSTM cells."""
        """Self defined functions """
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            # when a cell is constructed, we will need to use the mechanism called wrapper
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True
        )

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        # import pdb;pdb.set_trace()
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self, config):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            """ opaque_params,
                num_layers,
                num_units,
                input_size,
                input_mode=CUDNN_INPUT_LINEAR_MODE,
                direction=CUDNN_RNN_UNIDIRECTION,
                scope=None,
                name='cudnn_rnn_saveable'"""
            import pdb;pdb.set_trace()
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(
                        opaque_params = None,
                        num_layers=config.num_layers,
                        num_units=config.hidden_size,
                        input_size=config.hidden_size,
                        input_mode=CUDNN_INPUT_LINEAR_MODE,
                        direction=CUDNN_RNN_UNIDIRECTION,
                        scope="Model/RNN",
                        name='cudnn_rnn_saveable'
                        )
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
                self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
                self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def loss(self):
        return self._loss

    # @property
    # def grads(self):
    #     return self._grads
    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


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


##############################
# Training details
##############  Main Model Running ################
###### Note how do you run this in a remote servers#######
def main(_):
    # to increase the code robustness
    if not FLAGS.data_path:
        raise ValueError("Must set --")
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if (FLAGS.num_gpus > len(gpus)):
        raise ValueError("Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps  = 1

    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        initializer = tf.random_normal_initializer()
        with tf.name_scope("Train"):
            train_input = DetectorInput(config= config, frame_start=1, frame_end=260, name = "TrainInput")
            with tf.variable_scope("Model", reuse = None, initializer=initializer):
                m = VectModel(is_training = True, config = config, input_ = train_input)
            training_cost_sum = tf.summary.scalar("Training_Loss", m.cost)
            training_lr = tf.summary.scalar("Learning_Rate", m.lr)
            m.summary_merged = tf.summary.merge([training_lr, training_cost_sum])


        with tf.name_scope("Valid"):
            valid_input = DetectorInput(config= config, frame_start=261, frame_end = 280, name = "ValidInput")
            with tf.variable_scope("Model", reuse = True, initializer=initializer):
                mvalid = VectModel(is_training = False, config = config, input_ = valid_input)
            val_cost_sum = tf.summary.scalar("Validation_Loss", mvalid.cost)
            mvalid.summary_merged = val_cost_sum

        with tf.name_scope("Test"):# used to share variables
            test_input = DetectorInput(config= config, frame_start=281, frame_end = 301, name = "TestInput")
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
