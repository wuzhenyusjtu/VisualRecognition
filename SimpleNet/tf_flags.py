import tensorflow as tf

flags = tf.app.flags

'************************************************************************'
# You are runing risk to modify the parameter in this block
# Basic model parameters as external flags.
# Constants describing the training process.
flags.DEFINE_float('initial_learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average')
flags.DEFINE_float('num_epochs_per_decay', 10.0, 'Epochs after which learning rate decays')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Learning rate decay factor')
flags.DEFINE_integer('num_examples_per_epoch_for_train', 20000, 'Number of examples per epoch for train')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
'************************************************************************'

'************************************************************************'
# These parameters in this block can be modified without harm
flags.DEFINE_integer('max_steps_train', 2500, 'Number of batches to run for training.')
flags.DEFINE_integer('max_steps_test', 50, 'Number of batches to run for testing')

flags.DEFINE_string('model_dir', 'tmp/model', 'Directory where to write model proto to import in c++')
flags.DEFINE_string('train_dir', 'tmp/log', 'Directory where to write event logs and checkpoint.')
flags.DEFINE_string('records_directory', 'tmp/tfrecords', 'Directory to write the converted tfrecords file from raw images')
flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')


flags.DEFINE_string('checkpoint_dir', 'tmp/ckpt', 'Directory where to read model checkpoints.')
flags.DEFINE_float('validation_split_rate', 0.1, 'Rate of total images to separate from the original data set for the validation set.')
flags.DEFINE_float('testing_split_rate', 0.1, 'Rate of total images to separate from the original data set for the testing set')
flags.DEFINE_integer('num_classes', 5, 'Number of classes to do classification')
'************************************************************************'

FLAGS = flags.FLAGS

