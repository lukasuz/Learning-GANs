""" Models from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
"""
from keras.layers import Input, Embedding, Dense, Reshape, Conv2D, LeakyReLU, Concatenate, Flatten, Dropout, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

def calc_decay(lr, epoch=100):
    return lr/epoch

# define the standalone discriminator model
def discriminator(input_dim=(28,28,1), depth=64, n_classes=10, lr=0.0002):

    # Create embedding vector for conditional input, pipe through
    # linear dense layer and reshape to o.g. size
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = input_dim[0] * input_dim[1]
    li = Dense(n_nodes)(li)
    li = Reshape((input_dim[0], input_dim[1], 1))(li)

    # Concatenate and everything as usual
    in_image = Input(shape=input_dim)
    merge = Concatenate()([in_image, li])
    fe = Conv2D(depth, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(depth, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)

    out_layer = Dense(1, activation='sigmoid')(fe)
    model = Model([in_image, in_label], out_layer, name="Conditional_Discriminator")
    
    opt = Adam(lr=lr, beta_1=0.5, decay=calc_decay(lr))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def generator(input_dim=100, depth=64, n_classes=10):

    # Same as in the discriminator, but output size adapted
	in_label = Input(shape=(1,))
	li = Embedding(n_classes, 50)(in_label)
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	li = Reshape((7, 7, 1))(li)

	in_lat = Input(shape=(input_dim,))
	n_nodes = depth * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, depth))(gen)

	merge = Concatenate()([gen, li])
	gen = Conv2DTranspose(depth, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Conv2DTranspose(depth, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	model = Model([in_lat, in_label], out_layer, name="Conditional_Generator")
	return model

def gan(discriminator, generator, lr=0.0002):
	discriminator.trainable = False
	gen_noise, gen_label = generator.input
	gen_output = generator.output
	gan_output = discriminator([gen_output, gen_label])
	model = Model([gen_noise, gen_label], gan_output)
	opt = Adam(lr=lr, beta_1=0.5, decay=calc_decay(lr))
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