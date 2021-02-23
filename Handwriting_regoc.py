# this program classifies the MNIST handwritten digit images

import numpy as np
import mnist  # The Data set
import matplotlib.pyplot as plt
from keras.models import Sequential  # Ann architecture
from keras.layers import Dense  # Will provide the layers for ann
from keras.utils import to_categorical

# Loading the data set

train_images = mnist.train_images()  # Training data images
train_labels = mnist.train_labels()  # Training data labels
test_images = mnist.test_images()  # Training data images
test_labels = mnist.test_labels()  # Training data labels

# Normalizing the images. Normalizing from [0, 255] to [-0.5, 05]
# This is to make the ann easier to train

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
# Flatten the images. I am going to flatten each 28 * 28 image into a vector
# 28^2 = 784, This is the dimensional vector
# To pass into the ann

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Print the shape of the images

print(train_images.shape)  # (60000(rows), 784(columns) )

print(test_images.shape)  # (10000(rows), 784(columns) )

# Building the ann model
#  layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax', ))

# compiling the model
# needs a loss and optimizer function
# the loss function measures how well the model did on training
# and the tries to improve it by using the optimizer

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # allows us to use more than two classes
    metrics=['accuracy']
)

#  Train the model

model.fit(
    test_images,
    to_categorical(test_labels),
    batch_size=32,
    epochs=5
)

# Evaluate the model

model.evaluate(
    test_images,
    to_categorical(test_labels)
)

#  predictions on the first 5 teat images

predictions = model.predict(test_images)

# print the models predictions
print(np.argmax(predictions, axis=1))

print(test_labels)
#
for i in range(0, 100):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


