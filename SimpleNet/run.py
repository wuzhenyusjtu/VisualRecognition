import os.path
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from six.moves import xrange

import input_data
import convnet_model
from tf_flags import FLAGS
from classification_performance_evaluator import classification_performance_evaluator

display_step = 1
validation_step = 5
save_step = 50
IMAGE_PIXELS = 128 * 128 * 3

'''
function placeholder_inputs(...), train(...) are referenced from
https://github.com/HamedMP/ImageFlow/blob/master/example_project/my_cifar_train.py
'''
def placeholder_inputs(batch_size):
	"""Generate placeholder variables to represent the the input tensors.
	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the downloaded ckpt in the .run() loop, below.
	Args:
		batch_size: The batch size will be baked into both placeholders.
	Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
	"""
	# Note that the shapes of the placeholders match the shapes of the full
	# image and label tensors, except the first dimension is now batch_size
	# rather than the full size of the train or test ckpt sets.
	# batch_size = -1
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
	dropout_placeholder = tf.placeholder(tf.float32)
	return images_placeholder, labels_placeholder, dropout_placeholder


def train(continue_from_pre=True):
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)
		images_placeholder, labels_placeholder, dropout_placeholder = placeholder_inputs(FLAGS.batch_size)

		# Get images and labels by shuffled batch
		tr_images, tr_labels = input_data.inputs(filename='./tmp/tfrecords/train.tfrecords', batch_size=FLAGS.batch_size,
											num_epochs=FLAGS.num_epochs, num_threads=5, imshape=[128, 128, 3], use_distortion=False)
		val_images, val_labels = input_data.inputs(filename='./tmp/tfrecords/validation.tfrecords', batch_size=FLAGS.batch_size,
											num_epochs=FLAGS.num_epochs, num_threads=5, imshape=[128, 128, 3], use_distortion=False)
		# Build a Graph that computes the logits predictions from the inference model.
		logits = convnet_model.inference(images_placeholder, dropout_placeholder)
		# Calculate loss.
		loss = convnet_model.loss(logits, labels_placeholder)
		
		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnet_model.training(loss, global_step)
		
		# Calculate accuracy #
		acc, n_correct = convnet_model.evaluation(logits, labels_placeholder)
		
		# Create a saver.
		saver = tf.train.Saver()
		
		tf.scalar_summary('Training Acc', acc)
		tf.scalar_summary('Training Loss', loss)
		tf.image_summary('Training Images', tf.reshape(tr_images, shape=[-1, 128, 128, 3]), max_images=20)
		tf.image_summary('Validation Images', tf.reshape(val_images, shape=[-1, 128, 128, 3]), max_images=20)
		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.merge_all_summaries()
		
		# Build an initialization operation to run below.
		init = tf.initialize_all_variables()
		# Start running operations on the Graph.
		# NUM_CORES = 2  # Choose how many cores to use.
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, ))
		# inter_op_parallelism_threads=NUM_CORES,
		# intra_op_parallelism_threads=NUM_CORES))
		sess.run(init)
	
		# Write all terminal output results here
		output_f = open("tmp/output.txt", "ab")
		tr_f = open("tmp/training_accuracy.txt", "ab")
		val_f = open("tmp/validation_accuracy.txt", "ab")
		# Start the queue runners.ValueError: No variables to save

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
		
		if continue_from_pre:
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
			print ckpt.model_checkpoint_path
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print 'Session Restored!'
		try:
			while not coord.should_stop():
				for step in xrange(FLAGS.max_steps_train):
					tr_images_r, tr_labels_r = sess.run([tr_images, tr_labels])
					print tr_images_r.shape
					print tr_labels_r.shape
					val_images_r, val_labels_r = sess.run([val_images, val_labels])

					# Feed operation for training and validation
					tr_feed = {images_placeholder: tr_images_r, labels_placeholder: tr_labels_r, dropout_placeholder: 0.5}
					val_feed = {images_placeholder: val_images_r, labels_placeholder: val_labels_r, dropout_placeholder: 1.0}
					start_time = time.time()
					_, loss_value = sess.run([train_op, loss], feed_dict=tr_feed)
					duration = time.time() - start_time
					assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
					if step % display_step == 0:
						num_examples_per_step = FLAGS.batch_size
						examples_per_sec = num_examples_per_step / duration
						sec_per_batch = float(duration)
						format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f sec/batch)')
						print_str_loss = format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch)
						print (print_str_loss)
						output_f.write(print_str_loss + '\n')
						summary_str = sess.run([summary_op], feed_dict=tr_feed)
						summary_writer.add_summary(summary_str[0], step)

					if step % validation_step == 0:
						tr_acc, tr_n_correct = sess.run([acc, n_correct], feed_dict=tr_feed)
						format_str = '%s: step %d,  training accuracy = %.2f, n_correct= %d'
						print_str = format_str % (datetime.now(), step, tr_acc, tr_n_correct)
						output_f.write(print_str + '\n')
						tr_f.write(str(tr_acc) + '\n')
						print (print_str)
						val_acc, val_n_correct = sess.run([acc, n_correct], feed_dict=val_feed)
						format_str = '%s: step %d,  validation accuracy = %.2f, n_correct= %d'
						print_str = format_str % (datetime.now(), step, val_acc, val_n_correct)
						output_f.write(print_str + '\n')
						val_f.write(str(val_acc) + '\n')
						print (print_str)

					if step % save_step == 0 or (step+1) == FLAGS.max_steps_train:
						checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
						saver.save(sess, checkpoint_path, global_step=step)
		except tf.errors.OutOfRangeError:
			print ('Done traning -- epoch limit')
		finally:
			output_f.write('********************Finish Training********************')
			tr_f.write('********************Finish Training********************')
			val_f.write('********************Finish Training********************')
			coord.request_stop()
		
		coord.join(threads)
		sess.close()

def test():
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)

		images_placeholder, labels_placeholder, dropout_placeholder = placeholder_inputs(FLAGS.batch_size)
		images, labels = input_data.inputs(filename='./tmp/tfrecords/test.tfrecords', batch_size=FLAGS.batch_size,
											num_epochs=FLAGS.num_epochs, num_threads=5, imshape=[128, 128, 3])		

		logits = convnet_model.inference(images_placeholder, dropout_placeholder)
		acc, predictions_with_labels = convnet_model.testing(logits, labels_placeholder)
		saver = tf.train.Saver()
		init = tf.initialize_all_variables()
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, ))
		sess.run(init)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
		print ckpt.model_checkpoint_path
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Restored!')

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		# The order of category names here does not matter
		names_categories = ['adafor', 'brkt2', 'brng', 'fl3', 'flng']
		evaluator = classification_performance_evaluator(names_categories)
		while not coord.should_stop():

			for step in xrange(FLAGS.max_steps_test):
				te_images, te_labels = sess.run([images, labels])
				print te_images.shape
				print te_labels.shape
				te_feed = {images_placeholder: te_images, labels_placeholder: te_labels, dropout_placeholder: 1.0}
				te_acc, te_predictions_with_labels = sess.run([acc, predictions_with_labels], feed_dict=te_feed)
				print ('Step ' + str(step) + ' Testing Accuracy: ' + str(te_acc))
				predictions = te_predictions_with_labels[0]
				_labels = te_predictions_with_labels[1]
				print _labels
				evaluator.update(_labels, predictions)
			coord.request_stop()

		sess.close()
		evaluator.print_performance()
		
def main(argv=None):
	train(continue_from_pre=False)
	#test()

if __name__ == '__main__':
	tf.app.run()