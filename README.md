# Cat vs Dog Image Classification

## Overview

This project aims to classify images of cats and dogs using a convolutional neural network (CNN) built with TensorFlow 2.0 and Keras. The goal is to achieve a classification accuracy of at least 63%, with extra credit for reaching 70% accuracy. The dataset consists of images organized into training, validation, and test directories.

## Project Structure

The dataset directory structure is as follows:
```
cats_and_dogs
|__ train:
|______ cats: [cat.0.jpg, cat.1.jpg ...]
|______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
|______ cats: [cat.2000.jpg, cat.2001.jpg ...]
|______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```

## Requirements

- TensorFlow 2.0
- Keras
- NumPy
- Matplotlib

You can install the necessary packages using pip:

```bash
pip install tensorflow numpy matplotlib
```
## Instructions
**Cell 1: Import Libraries**

Make sure to import the required libraries. The necessary imports are TensorFlow, Keras, NumPy, and Matplotlib.
**Cell 2: Download Data and Set Key Variables**

Download the dataset and set the key variables for image dimensions, batch size, and directory paths.
**Cell 3: Create Image Generators**

Set up image generators for training, validation, and test datasets using ImageDataGenerator. Use the rescale argument to normalize pixel values between 0 and 1.

```python

# Example code for setting up image generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_data_gen = train_datagen.flow_from_directory(
    directory='cats_and_dogs/train',
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data_gen = validation_datagen.flow_from_directory(
    directory='cats_and_dogs/validation',
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data_gen = test_datagen.flow_from_directory(
    directory='cats_and_dogs/test',
    batch_size=1,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=None,
    shuffle=False
)
```
**Cell 4: Plot Images**

Use the plotImages function to visualize five random training images. This function is already provided.
**Cell 5: Data Augmentation**

Recreate the train_image_generator using ImageDataGenerator with additional random transformations to prevent overfitting. Include 4-6 transformations like rotation, width shift, height shift, shear, zoom, and horizontal flip.

```python

# Example code for data augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
**Cell 6: Augmented Images**

Visualize a single image plotted with five different variations using the updated train_image_generator.
**Cell 7: Build Model**

Create a CNN model using the Keras Sequential API. Include Conv2D and MaxPooling2D layers, followed by a fully connected layer with a ReLU activation function. Compile the model with an optimizer, loss function, and accuracy metric.

```python

# Example code for building the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
**Cell 8: Train Model**

Train the model using the fit method. Pass in the training data, validation data, and other necessary parameters such as epochs and steps per epoch.

```python

# Example code for training the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // batch_size
)
```
**Cell 9: Visualize Training Results**

Run the provided code to visualize the accuracy and loss of the model during training.
**Cell 10: Predict Test Images**

Use the trained model to predict the class of each test image. Get the probabilities and plot the test images with their predicted class probabilities.

```python

# Example code for making predictions
test_predictions = model.predict(test_data_gen, verbose=1)
```
**Cell 11: Challenge Completion**

Run the final cell to check if you have successfully completed the challenge and achieved the desired accuracy.
License

This project is licensed under the MIT License. See the LICENSE file for details.

```css


Feel free to adjust the details or code snippets based on your specific implementation!
