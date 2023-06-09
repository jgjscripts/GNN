import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to the range of 0 to 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical one-hot encoding
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Function to convert images and labels to Example protos
def image_example(image, label):
    feature = {
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Function to write the TFRecords file
def write_tfrecord(tfrecord_file, images, labels):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(images.shape[0]):
            image = images[i]
            label = labels[i]
            example = image_example(image, label)
            writer.write(example.SerializeToString())

# Write the training and test data to TFRecords files
write_tfrecord('train.tfrecords', train_images, train_labels)
write_tfrecord('test.tfrecords', test_images, test_labels)

# Function to parse the Example protos
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([32*32*3], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.reshape(example['image'], (32, 32, 3))
    label = example['label']
    return image, label

# Create training and test datasets
train_dataset = tf.data.TFRecordDataset('train.tfrecords').map(parse_tfrecord_fn)
test_dataset = tf.data.TFRecordDataset('test.tfrecords').map(parse_tfrecord_fn)

# Normalize pixel values to the range of 0 to 1
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

# Shuffle and batch the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(50000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Create an instance of the residual attention network
model = residual_attention_network((32, 32, 3), num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_dataset, epochs=epochs)

# Evaluate the model
model.evaluate(test_dataset)



##########################################################################################################################################
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to the range of 0 to 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical one-hot encoding
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Function to convert images and labels to Example protos
def image_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Function to write the TFRecords file
def write_tfrecord(tfrecord_file, images, labels):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(images.shape[0]):
            image = images[i]
            label = labels[i]
            example = image_example(image, label)
            writer.write(example.SerializeToString())

# Write the training and test data to TFRecords files
write_tfrecord('train.tfrecords', train_images, train_labels.argmax(axis=1))
write_tfrecord('test.tfrecords', test_images, test_labels.argmax(axis=1))

# Function to parse the Example protos
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example['label']
    return image, label

# Function to preprocess the images and labels
def preprocess_fn(image, label):
    # Apply any preprocessing steps here (e.g., data augmentation)
    image = tf.image.resize(image, [224, 224])  # Resize image to desired size
    return image, label

# Function to create the dataset
def create_dataset(tfrecord_file, batch_size, preprocess_fn):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create training and test datasets
train_dataset = create_dataset('train.tfrecords', batch_size=32, preprocess_fn=preprocess_fn)
test_dataset = create_dataset('test.tfrecords', batch_size=32, preprocess_fn=preprocess_fn)

# Create an instance of the residual attention network
input_shape = (224, 224, 3)
num_classes = 10
model = residual_attention_network(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Evaluate the model
model.evaluate(test_dataset)

