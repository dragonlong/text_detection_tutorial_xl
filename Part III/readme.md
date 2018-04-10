## This is a implementation of both GAP-EAST CNN model and Video-Text LSTM model
- 1. GAP-EAST trains heat-map representation on ICDAR2015 data(about 1000 images)
- 2. Then use the benchmark scripts generate feature data from icdar video dataset
- 3. Modify the data path and checkpoints path, run train_lstm_2 
Notes: both tensorboard and pyplot visualization are integrated, so it's recommended to
train locally first, so you could observe both curve and images during training. When training on remote cluster
visualization could be realized by launching tensorboard and ssh
