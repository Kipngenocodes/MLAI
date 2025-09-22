import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tf_keras  # Add this import

# Downloading the IMDB dataset
# Split the train set into 60% training and 40% validation to end up with 15,000 examples 
# for training, 10,000 for validation and 25,000 for testing
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Explore the data needed for training
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
  
# Building the model  that works with data in string format
# Using a pre-trained text embedding model from TensorFlow Hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                        dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# Building the model with tf_keras
model = tf_keras.Sequential()  # Use tf_keras instead of tf.keras
model.add(hub_layer)           # This should now work
model.add(tf_keras.layers.Dense(16, activation='relu'))
model.add(tf_keras.layers.Dense(1))  # For binary classification (sigmoid is implied later)

model.summary()  # Print model summary

# Compiling the model
# loss function and optimizer
model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

# Train the model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)


# Evaluate the model
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))