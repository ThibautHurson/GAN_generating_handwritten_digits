import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def get_random_vect(size, nb_samples):
    vect = np.random.randn(nb_samples, size)
    return vect

def get_fake_images(generator, random_vect_size, nb_samples):
    rand_vect = get_random_vect(random_vect_size, nb_samples)
    x_gen = generator.predict(rand_vect)
    
    # Create the label vector for these fake images
    y_gen = np.zeros(nb_samples)
    
    return x_gen, y_gen

def predict_and_plot(generator, random_vect_size):
    x_gen, _ = get_fake_images(generator, random_vect_size, 100)
    x_gen = np.squeeze(x_gen)

    # Plot some image examples
    plt.figure(figsize=(10,10))
    i = 0
    for i in range (100):
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(x_gen[i], cmap=plt.cm.binary)
        i += 1

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If one or more GPUs are available, we run on GPU
if len(tf.config.list_physical_devices('GPU')):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

# cpus = tf.config.experimental.list_physical_devices('CPU')
# tf.config.experimental.set_memory_growth(cpus[0], True)
filename = 'generator_model_022.h5'
predict_and_plot(load_model(filename), random_vect_size=100)
plt.show()