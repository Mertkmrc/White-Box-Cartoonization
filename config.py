import tensorflow as tf
from tensorflow import keras



IMAGE_SIZE = 256

def transform_train(image):
    image = keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(image)
    image = keras.layers.RandomFlip("horizontal")(image)
    image = keras.layers.Rescaling(1./255)(image)
    return image

transform_test = keras.Sequential([
    keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    keras.layers.Rescaling(1./255),
])