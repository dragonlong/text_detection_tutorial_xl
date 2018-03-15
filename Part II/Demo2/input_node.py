import utils.reader as reader


class DetectorInput(object):
    """The input data."""
    def __init__(self, config, frame_start, frame_end, FLAGS, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (((frame_end- frame_start) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.vect_producer(FLAGS.data_path, frame_start, frame_end, batch_size, num_steps, name=name)
