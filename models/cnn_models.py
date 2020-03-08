"""
Models from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
"""

import keras
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Activation, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def discriminator(input_dim=(28,28,1), depth=32, dropout=0.4, lr=0.00004):
    model = Sequential()
	
    model.add(Conv2D(int(depth/2), (3,3), strides=(2,2), padding='same', input_shape=input_dim))
    model.add(LeakyReLU(alpha=0.2))
	
    model.add(Conv2D(depth, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
	
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
	
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    return model

def generator(input_dim=100, depth=32, dim=7):
    model = Sequential()
    n_nodes = depth * 7 * 7
    model.add(Dense(n_nodes, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, depth)))

    model.add(Conv2DTranspose(int(depth/2), (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(int(depth/2), (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model

def gan(discriminator, generator, gen_input_dim=100, lr=0.00002):
	discriminator.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	opt = Adam(lr=lr, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


if __name__ == "__main__":
    print("Generator:")
    generator = generator()
    generator.summary()
    print('\n')

    print("Discriminator:")
    discriminator = discriminator()
    discriminator.summary()
    print('\n')

    print("GAN:")
    gan = gan(discriminator, generator)
    gan.summary()
    print('\n')