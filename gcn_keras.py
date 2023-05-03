pip install tensorflow keras numpy matplotlib networkx

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the input images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define graph structure
G = nx.grid_2d_graph(32, 32)

# Compute adjacency matrix
adj_matrix = nx.adjacency_matrix(G)

# Define GCNN model
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Reshape((8, 8))(x)
x = layers.Dense(10, activation="softmax")(x)
outputs = layers.GraphConvolution(adj_matrix)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Compile and train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Plot accuracy and loss curves
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.show()

# In the above code, we first load the CIFAR-10 dataset and normalize the input images. We also convert the labels to one-hot encoding. We then define the graph structure using the nx.grid_2d_graph function from the NetworkX library. We compute the adjacency matrix using the nx.adjacency_matrix function.

# We then define the GCNN model using the Keras Functional API. The model consists of two convolutional layers followed by two max-pooling layers, a dense layer, and a reshape layer. The output of the reshape layer is passed through the GraphConvolution layer, which applies the graph convolution operation on the input feature maps using the adjacency matrix.

# We compile and train the model using the compile and fit methods of the model object. Finally, we evaluate the model on the test set and plot the accuracy and loss curves using Matplotlib.

