
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to the directory containing the images
train_dir = '/path/to/training/images/'
val_dir = '/path/to/validation/images/'
test_dir = '/path/to/test/images/'

# Set the batch size and image dimensions
batch_size = 32
img_height = 256
img_width = 256

# Create the data generators for the training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width ,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Make predictions on new images
new_images = ['rajasthan_nighttime_lights_2015_7.tif', 'rajasthan_nighttime_lights_2015_8.tif']
new_images_dir = '/path/to/new/images/'
new_images_data = np.array([tf.keras.preprocessing.image.load_img(os.path.join(new_images_dir, img), target_size=(img_height, img_width)) for img in new_images])
new_images_data = new_images_data.reshape((-1, img_height, img_width, 3)) / 255.
predictions = model.predict(new_images_data)

