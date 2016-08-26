import os
import tensorflow as tf
from tf_flags import FLAGS
from random import shuffle
import numpy as np
import glob

'''Python image library PIL can be installed by "conda install PIL"''' 
from PIL import Image
import gc
import sys

'''
function: _int64_feature, _bytes_feature, convert_to_records are referenced at
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
'''

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""Convert the images with labels into readable file format for the ConvNet in tensorflow"""
"""Converts images data to TFRecords file format with Example protos."""
def convert_to_records(images, labels, name):
	num_examples = labels.shape[0]
	if images.shape[0] != num_examples:
		raise ValueError("Images size %d does not match label size %d." %
												(images.shape[0], num_examples))
	rows = images.shape[1]
	cols = images.shape[2]
	depth = images.shape[3]
	
	filename = os.path.join(FLAGS.records_directory, name + '.tfrecords')
	print('Writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	for index in range(num_examples):
		image_raw = images[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
				'height': _int64_feature(rows),
				'width': _int64_feature(cols),
				'depth': _int64_feature(depth),
				'label': _int64_feature(int(labels[index])),
				'image_raw': _bytes_feature(image_raw)}))
		writer.write(example.SerializeToString())

def crop_resize_image(image, crop_width, crop_height, resize_width, resize_height):
	# Get dimensions
	width, height = image.size
	left = (width - crop_width)/2
	top = (height - crop_height)/2
	right = (width + crop_width)/2
	bottom = (height + crop_height)/2
	image = image.crop((left, top, right, bottom))
	image = image.resize((resize_width, resize_height), Image.ANTIALIAS)
	#gc.collect()
	return image

def read_images_labels_from(path, name_categories):
	images = []
	labels = []
	num_categories = len(name_categories)
	count_lst = [0] * num_categories
	#files_path = glob.glob(os.path.join(path, 'train/', '{*.[pP][nN][gG],*.[jJ][pP][eE][gG], *.[jJ][pP][gG]}'))
	files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
	files_path += glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
	files_path += glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
	
	i = 0
	for filename in files_path:
		# .convert("L")  # Convert to greyscale
		im = Image.open(filename)  
		width, height = im.size
		# To create a square image
		if width > height:
			im = crop_resize_image(im, height, height, 128, 128)
		else:
			im = crop_resize_image(im, width, width, 128, 128)
		im = np.asarray(im, np.uint8)
		# get only images name, not path
		# convert the image name to lower case
		image_name = filename.split('/')[-1].split('.')[0].lower()

		filename_matched_category = False
		for index in xrange(num_categories):
			if image_name.find(name_categories[index]) != -1:
				labels.append(index)
				images.append(im)
				count_lst[index] += 1
				filename_matched_category = True
				break
		if not filename_matched_category:
			print image_name
			raise ValueError('filename unable to match with categories!')
		i += 1
		if i%100==0:
			print str(i) + ' images read'
			print 'shape of a single image' + str(im.shape)
			print 'size of a single image: ' + str(sys.getsizeof(im))
			print 'size of a list: ' + str(len(images) * sys.getsizeof(images[0]))


	for index in xrange(num_categories):
		print 'number of ' + name_categories[index] + ': '+ str(count_lst[index])
	print 'total number of images: ' + str(sum(count_lst))
	
	_images = [np.asarray(image, np.uint8) for image in images]
	_images = np.array(_images)
	print 'images shape after flatten into array: ' +str(_images.shape)
	_labels = np.array(labels, dtype=np.uint32)
	return _images, _labels
	
def split_images_to_sets(images, labels):
	assert images.shape[0] == labels.shape[0], "Number of images, %d should be equal to number of labels %d" % \
											 (images.shape[0], labels.shape[0])
	
	# Here are some tricks to introduce randomness when we creating data set
	index_shuffle = range(len(images))
	images_shuffle = []
	labels_shuffle = []
	shuffle(index_shuffle)
	for i in index_shuffle:
		images_shuffle.append(images[i])
		labels_shuffle.append(labels[i])

	images_shuffle = np.array(images_shuffle)
	labels_shuffle = np.array(labels_shuffle, dtype=np.uint32)
	print 'Labels after shuffle' + str(labels)
	print labels_shuffle.shape
	print images_shuffle.shape
	valset_size = int(images_shuffle.shape[0] * FLAGS.validation_split_rate) 
	teset_size = int(images_shuffle.shape[0] * FLAGS.testing_split_rate)

	# Generate a validation set.
	te_images = images_shuffle[:teset_size, :, :, :]
	te_labels = labels_shuffle[:teset_size]
	val_images = images_shuffle[teset_size:(teset_size + valset_size), :, :, :]
	val_labels = labels_shuffle[teset_size:(teset_size + valset_size)]
	tr_images = images_shuffle[(teset_size + valset_size):, :, :, :]
	tr_labels = labels_shuffle[(teset_size + valset_size):]

	# Convert to Examples and write the result to TFRecords.
	convert_to_records(tr_images, tr_labels, 'train')
	convert_to_records(val_images, val_labels, 'validation')
	convert_to_records(te_images, te_labels, 'test')
	
def main(argv):
	# If you add new categories, need to change it here
	names_categories = ['adafor', 'brkt2', 'brng', 'fl3', 'flng']
	# The source of input images
	images, labels = read_images_labels_from('./raw_images/', names_categories)
	# Split the data into 3 different set, for training, validation and testing
	split_images_to_sets(images, labels)
	
if __name__ == '__main__':
	tf.app.run()