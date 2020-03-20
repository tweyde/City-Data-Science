# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Executes model training"""

import os.path
import logging
import tensorflow as tf

from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE

def get_batched_dataset(filenames, train=False):
  dataset = load_dataset(filenames)
  dataset = dataset.cache() # This dataset fits in RAM
  if train:
    # Best practices for Keras:
    # Training dataset: repeat then batch
    # Evaluation dataset: do not repeat
    dataset = dataset.repeat()
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
  # should shuffle too but this dataset was well shuffled on disk already
  return dataset
  # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets



def main():
    GCS_PATTERN = 'gs://flowers-public/tfrecords-jpeg-192x192-2/*.tfrec'
    IMAGE_SIZE = [192, 192]

    BATCH_SIZE = 64  # 128 works on GPU too but comes very close to the memory limit of the Colab GPU
    EPOCHS = 2

    VALIDATION_SPLIT = 0.19
    CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers',
               'tulips']  # do not change, maps to the labels in the data (folder names)

    # splitting data files between training and validation
    filenames = tf.io.gfile.glob(GCS_PATTERN)
    split = int(len(filenames) * VALIDATION_SPLIT)
    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]
    print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(
        len(filenames), len(training_filenames), len(validation_filenames)))
    validation_steps = int(3670 // len(filenames) * len(validation_filenames)) // BATCH_SIZE
    steps_per_epoch = int(3670 // len(filenames) * len(training_filenames)) // BATCH_SIZE
    print(
        "With a batch size of {}, there will be {} batches per training epoch and {} batch(es) per validation run.".format(
            BATCH_SIZE, steps_per_epoch, validation_steps))

    logging.basicConfig()

    pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
    # pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    # pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    # pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # instantiate the datasets
    training_dataset = get_batched_dataset(training_filenames, train=True)
    validation_dataset = get_batched_dataset(validation_filenames, train=False)

    history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        validation_data=validation_dataset, validation_steps=validation_steps)
    # The model name should remain 'model.joblib' for
    # AI Platform to be able to create a model version.
    model_name = os.path.join(sys.argv[1], 'model.joblib')
    logging.info('Model will be saved to "%s..."', model_name)
    temp_file = '/tmp/model.joblib'
    joblib.dump(model, temp_file)

    # Copy the temporary model file to its destination
    with tf.io.gfile.GFile(temp_file, 'rb') as temp_file_object:
        with tf.io.gfile.GFile(model_name, 'wb') as file_object:
            file_object.write(temp_file_object.read())

    logging.info('Model was saved')


if __name__ == '__main__':
    main()
