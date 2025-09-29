import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# Download the IMDB dataset
# Split the train set into 60% training and 40% validation to end up with 15,000 examples 
# for training, 10,000 for validation and 25,000 for testing
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True  
)
