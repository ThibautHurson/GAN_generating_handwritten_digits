# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.utils import shuffle

# Load the mnist dataset
from tensorflow.keras.datasets.mnist import load_data
from timeit import default_timer as timer   

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If one or more GPUs are available, we run on GPU
if len(tf.config.list_physical_devices('GPU')):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

start = timer()

(x_train, y_train), (x_test, y_test) = load_data()
print('Train shape:', x_train.shape, y_train.shape)
print('Test shape:', x_test.shape, y_test.shape)

# Reshape images
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))

# Normalize
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

def get_discriminator(image_shape=(28,28,1)):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    
    model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    # model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0002,beta_1=0.5),
                  metrics=['accuracy'])
    return model


def get_generator(random_vect_size):
    model = Sequential()
    
    # Base image of shape 7x7
    model.add(Dense(512*7*7, activation='relu', input_dim=random_vect_size))
    model.add(Reshape((7, 7, 512)))
    
    # Upsample to 14x14
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), activation='relu', padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    
    # Upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), activation='relu', padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))

    # Output layer
    model.add(Conv2DTranspose(1, (7,7), activation='tanh', padding='same'))
    
    return model


def get_random_vect(size, nb_samples):
    vect = np.random.randn(nb_samples, size)
    return vect

def get_fake_images(generator, random_vect_size, nb_samples):
    rand_vect = get_random_vect(random_vect_size, nb_samples)
    x_gen = generator.predict(rand_vect)
    
    # Create the label vector for these fake images
    y_gen = np.zeros(nb_samples)
    
    return x_gen, y_gen

def get_GAN(generator, discriminator):
    discriminator.trainable = False
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0002,beta_1=0.5),
                  metrics=['accuracy'])
    return model

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


def train(generator, discriminator, gan, x_train, random_vect_size, epochs=3, batch=128):
    # Number of batches per epoch
    ba_per_ep = int(x_train.shape[0]/batch)

    for i in range(epochs):
        x_train = shuffle(x_train)
        for j in range(ba_per_ep):
            # generate real samples
            x_real = x_train[batch*j: batch*(j+1)]
            y_real = np.ones(x_real.shape[0])
            # generate fake samples
            x_fake, y_fake = get_fake_images(generator, random_vect_size, batch)            
            # Stack real and fake samples, and shuffle
            D_x_train, D_y_train = np.vstack((x_real, x_fake)), np.concatenate((y_real, y_fake))
            # Update the discriminator
            D_loss, _ = discriminator.train_on_batch(D_x_train, D_y_train)
            
            # Create the random inputs for the generator
            x_gen = get_random_vect(random_vect_size, batch)
            # Create inverted labels vector for these fake images that will be used to train the generator
            y_gen = np.ones(batch)
            # Update the generator
            G_loss, _ = gan.train_on_batch(x_gen, y_gen)
            G_loss, _ = gan.train_on_batch(x_gen, y_gen) #update twice to avoid the rapid convergence of the discriminator

            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, ba_per_ep, D_loss, G_loss))
        print('Epoch' + str(i) + ': ' + 'Gen loss ' + str(G_loss) + ', Dis loss ' + str(D_loss))
        # save the generator model tile file
        filename = 'generator_model_%03d.h5' % (i + 1)
        generator.save(filename)
        predict_and_plot(load_model(filename), random_vect_size)
    plt.show()



random_vect_size = 100
discriminator = get_discriminator()
generator = get_generator(random_vect_size)
gan = get_GAN(generator, discriminator)
train(generator, discriminator, gan, x_train, random_vect_size, epochs=35,  batch=128)

print("Total time:", timer()-start)
