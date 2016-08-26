import os
import numpy as np
import tensorflow as tf

def _dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arrange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    print(labels_one_hot[0])
    return labels_one_hot

'''
function read_and_decode(...) are referenced from
https://github.com/HamedMP/ImageFlow/blob/master/imageflow/reader.py
'''
def read_and_decode(filename_queue, imshape, normalize=False, flatten=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
    })
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    if flatten:
        num_elements = 1
        for i in imshape:
            num_elements = num_elements * i
        print num_elements
        image = tf.reshape(image, [num_elements])
        image.set_shape(num_elements)
    else:
        image = tf.reshape(image, imshape)
        image.set_shape(imshape)
    if normalize:
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    return image, label

'''
function inputs(...), _generate_image_and_label_batch(...) are referenced from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_input.py
https://github.com/HamedMP/ImageFlow/blob/master/imageflow/imageflow.py
'''
   
def inputs(filename, batch_size, num_epochs, num_threads, imshape, num_examples_per_epoch=128, use_distortion=False):
    """Reads input tfrecord file num_epochs times. Use it for validation.
    Args:
        filename: The path to the .tfrecords file to be read
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input ckpt, or 0/None to
             train forever.
        num_threads: Number of reader workers to enqueue
        imshape: The shape of image in the format
        num_examples_per_epoch:
    Returns:
        Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size, with the true label of a number in the range [0, NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs:
        num_epochs = None
    if not tf.gfile.Exists(filename):
        raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs, name='string_input_producer')
        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue, imshape, normalize=True)

        if use_distortion:
            # Reshape to [32, 32, 3] as distortion methods need this shape
            image = tf.reshape(image, imshape)
            #image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
            #image = tf.image.resize_images(image, 32, 32)
            height = imshape[1]
            width = imshape[0]
            # Image processing for training the network. Note the random distortions applied to the image.
            # Removed random_crop in new TensorFlow release.
            # Randomly crop a [height, width] section of the image.
            # distorted_image = tf.image.resize_image_with_crop_or_pad(image, height, width)
            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(image)
            # Because these operations are not commutative, consider randomizing
            # randomize the order their operation.
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
            # # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_whitening(distorted_image)
            num_elements = 1
            for i in imshape:
                num_elements = num_elements * i
            image = tf.reshape(float_image, [num_elements])

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        #images, sparse_labels = _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, num_threads, shuffle=True)
        
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_queue_examples, name='batching_shuffling')
        
    return images, sparse_labels

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, num_preprocess_threads, shuffle):
    """Construct a queued batch of images and labels.
    Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            enqueue_many=False,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_queue_examples,
            name='batch_shuffling')
        print '****************shuffling****************'
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size, 
            enqueue_many=False,
            name='batch_unshuffling')
        print '****************unshuffling****************'

    return images, label_batch
