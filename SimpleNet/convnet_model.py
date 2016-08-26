import tensorflow as tf
from tf_flags import FLAGS
import re

TOWER_NAME = 'tower'


'''
2 Convolution + MaxPooling layers and 3 fully connected layers (including the output softmax layer)
all the functions here are referenced from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/alexnet/alexnet_benchmark.py
'''

def _activation_summary(x):
	"""Helper to create summaries for activations.
	Creates a summary that provides a histogram of activations.
	Creates a summary that measure the sparsity of activations.
	Args:
	x: Tensor
	Returns:
	nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
	"""Add summaries for losses.
	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.
	Args:
	total_loss: Total loss from loss().
	Returns:
	loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])
	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
	# Name each loss as '(raw)' and name the moving average version of the loss
	# as the original loss name.
		tf.scalar_summary(l.op.name +' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))
	return loss_averages_op
	
def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable
	Returns:
	Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var
	
def _variable_with_weight_decay(name, shape, stddev, wd):
	var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var
	
def inference(images, keep_prob):
	"""Build the model.
	Args:
	images: Images returned from distorted_inputs() or inputs().
	Returns:
	Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	# Reshape input picture
	print('In Inference ', images.get_shape(), type(images))
	images = tf.reshape(images, shape=[-1, 128, 128, 3])
	print images
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
										stddev=1e-4, wd=0.0)
		conv1 = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv1, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv1)
	# pool1
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						padding='SAME', name='pool1')
	print conv1
	# norm1
	#conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
	#					name='norm1')
	# dropout (keep probability)
	#conv1 = tf.nn.dropout(conv1, keep_prob)

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
										 stddev=1e-4, wd=0.0)
		conv2 = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv2, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv2)
		
	# norm2
	#conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
	#					name='norm2')
	# pool2
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
						strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	#conv2 = tf.nn.dropout(conv2, keep_prob)
	print conv2

	# local3
	with tf.variable_scope('local3') as scope:
	# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(conv2, [FLAGS.batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
										  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		#local3 = tf.nn.dropout(local3, keep_prob)  # Apply Dropout
		_activation_summary(local3)
	print local3

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
										  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)
	print local4

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, FLAGS.num_classes],
										  stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [FLAGS.num_classes],
							  tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	print softmax_linear

	return softmax_linear
	
def loss(logits, labels):
	"""Add L2Loss to all the trainable variables.
		Add summary for for "Loss" and "Loss/avg".
		Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
			of shape [batch_size]
		Returns:
		Loss tensor of type float.
	"""
	# Reshape the labels into a dense Tensor of
	# shape [batch_size, NUM_CLASSES].
	sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
	indices = tf.reshape(tf.range(0, FLAGS.batch_size), [FLAGS.batch_size, 1])
	concated = tf.concat(1, [indices, sparse_labels])
	dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, FLAGS.num_classes], 1.0, 0.0)
	#print('***********In Inference*********** ', logits.get_shape(), type(logits))
	#print('***********In Inference*********** ', dense_labels.get_shape(), type(dense_labels))
	# Calculate the average cross entropy loss across the batch.
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		logits, dense_labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def training(total_loss, global_step):
	"""Train the model.
	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.
	Args:
	total_loss: Total loss from loss().
	global_step: Integer Variable counting the number of training steps processed.
	Returns:
	train_op: op for training.
	"""
	# Variables that affect learning rate.
	num_batches_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

	print('Decay steps is: ', decay_steps)
	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
								  global_step,
								  decay_steps,
								  FLAGS.learning_rate_decay_factor,
								  staircase=True)
	tf.scalar_summary('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):

		train_op = tf.no_op(name='train')

	return train_op

def testing(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size], with values in the
			range [0, NUM_CLASSES).
	Returns:
		A 2D int32 tensor with predictions and corresponding labels.
	"""
	print('Testing...')
	correct = tf.nn.in_top_k(logits, labels, 1)
	predictions = tf.argmax(logits, 1)
	predictions = tf.cast(predictions, tf.int32)
	num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
	acc_percent = num_correct / FLAGS.batch_size
	return acc_percent * 100.0, tf.pack([predictions, labels])
	
def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size], with values in the
			range [0, NUM_CLASSES).
	Returns:
		A scalar int32 tensor with the number of examples (out of batch_size)
		that were predicted correctly.
	"""
	print('Evaluation...')
	# For a classifier model, we can use the in_top_k Op.
	# It returns a bool tensor with shape [batch_size] that is true for
	# the examples where the label's is was in the top k (here k=1)
	# of all logits for that example.
	correct = tf.nn.in_top_k(logits, labels, 1)
	num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
	acc_percent = num_correct / FLAGS.batch_size
	# Return the number of true entries.
	return acc_percent * 100.0, num_correct
