""" 
Models adapted from https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
"""

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.layers.advanced_activations import ReLU
from keras.optimizers import Adam

def discriminator(input_dim=784, lr=0.001):
    """ Dense Discriminator
    """
    m = Sequential()
    m.add(Dense(units=1024, input_dim=input_dim))
    m.add(ReLU())
    m.add(Dropout(0.3))
    
    m.add(Dense(units=512))
    m.add(ReLU())
    m.add(Dropout(0.3))
       
    m.add(Dense(units=256))
    m.add(ReLU())
    
    m.add(Dense(units=1, activation='sigmoid'))
    m.name = "Dense_Discriminator"

    m.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])
    
    return m

def generator(input_dim=100):
    """ Dense Generator.
    """
    m = Sequential()
    m.add(Dense(units=256, input_dim=input_dim))
    m.add(ReLU())
    m.add(Dropout(0.3))
    
    m.add(Dense(units=512))
    m.add(ReLU())
    m.add(Dropout(0.3))
    
    m.add(Dense(units=1024))
    m.add(ReLU())
    
    m.add(Dense(units=784, activation='tanh'))
    m.name = "Dense_Generator"

    # m.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    return m


def gan(discriminator, generator, gen_input_dim=100, lr=0.008):
    """ Dense GAN.
    """
    discriminator.trainable = False # Do not train generator during discriminator
    gan_input = Input(shape=(gen_input_dim,)) # Input noise 100 size vector
    x = generator(gan_input) # Input is noise vector
    gan_output = discriminator(x) # output is output of the generator
    gan = Model(inputs=gan_input, outputs=gan_output, name="Dense_GAN") # combine both to a unified model

    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.5), metrics=['accuracy'])

    return gan

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