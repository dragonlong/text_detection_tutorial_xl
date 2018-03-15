
import tensorflow as tf
###############################
# Config file
####################################################
global FLAGS
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION  = "unidirection"


def get_config(FLAGS):
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
    output_size   = 360
    hidden_size   = 200
    max_max_epoch = 500
    max_epoch     = 100
    keep_prob     = 1.0
    lr_decay      = 0.999
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
    lr_decay = 0.999
    batch_size = 20
    vocab_size = 6663
    rnn_mode = CUDNN
