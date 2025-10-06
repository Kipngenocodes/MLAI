import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Custom Keras layer to wrap hub.KerasLayer
class HubLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, hub_url, **kwargs):
        super(HubLayerWrapper, self).__init__(**kwargs)
        self.hub_layer = hub.KerasLayer(hub_url, input_shape=(), dtype=tf.string, trainable=True)
    
    def call(self, inputs):
        return self.hub_layer(inputs)

# Set TensorFlow Hub cache directory
cache_dir = "C:/Users/p1/tfhub_cache"
os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
os.environ["TFHUB_CACHE_DIR"] = cache_dir

# Print versions for debugging
print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorFlow Hub Version: {hub.__version__}")

# Load the IMDB reviews dataset
try:
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Explore a batch of training data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print("Examples:\n", train_examples_batch.numpy())
print("Labels:\n", train_labels_batch.numpy())

# Load pre-trained embedding layer
embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
try:
    hub_layer = HubLayerWrapper(embedding_url)
    print("Embedding layer loaded successfully")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    raise

# Test embedding layer
try:
    embeddings = hub_layer(train_examples_batch[:3])
    print("Embedding output shape:", embeddings.shape)
except Exception as e:
    print(f"Error testing embedding layer: {e}")
    raise

# Build the model using Functional API
inputs = tf.keras.Input(shape=(), dtype=tf.string)
x = hub_layer(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)  # Logits output for binary classification
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
try:
    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=10,
        validation_data=validation_data.batch(512),
        callbacks=[early_stopping],
        verbose=1
    )
    print("Training completed successfully")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Print training history
print("Training history:", history.history)

# Evaluate the model
try:
    results = model.evaluate(test_data.batch(512), verbose=2)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.3f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

# Save the model
try:
    save_dir = 'C:/Users/p1/models/imdb_model.keras'  # Added .keras extension
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # Ensure save directory exists
    model.save(save_dir)  # Save in native Keras format
    # For SavedModel format (e.g., for TFLite/TFServing), use:
    # model.export('C:/Users/p1/models/imdb_model_savedmodel')
    print("Model saved successfully")
except Exception as e:
    print(f"Error saving model: {e}")
    raise
