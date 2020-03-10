import math;
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras.backend as K

import models.dense_models as dense_models
import models.cnn_models as cnn_models

save_path = "./imgs/"
log_dir = "./logs/"
disc_input_dim_flat = 784
disc_input_dim = (28,28,1)
gen_input_dim = 100
model_type = "cnn" # cnn / dense / conditional

save_interval = 5
epochs = 400
batch_size = 256

def load_data(flatten=True):
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # flatten 28x28 -> 784
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

    return (x_train, y_train, x_test, y_test)


def plot_generated_images(epoch, generator, noise=None, examples=100, dim=(10,10),
                          figsize=(10,10), model_type=model_type):

    if noise == None:
        noise = np.random.normal(-1.0, 1.0, size=[examples, 100])

    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path+'%s_image %d.png' % (model_type, epoch))
    plt.close()

def init_write_logs(identifer, model_type=model_type, log_dir=log_dir):
    path = "{0}{1}_{2}.txt".format(log_dir, model_type, identifer)
    with open(path, 'w') as  f:
        f.write("Loss, Accuracy\n")


def append_line(loss, acc, identifer, model_type=model_type, log_dir=log_dir):
    path = "{0}{1}_{2}.txt".format(log_dir, model_type, identifer)
    with open(path, 'a') as  f:
        f.write("{0},{1}\n".format(loss, acc))


def train(discriminator, generator, gan, noise_input=None, load_data_func=load_data, 
          gen_input_dim=100, epochs=400, batch_size=256, save_interval=0):

        (x_train, y_train, _, _) = load_data_func()

        # Due to flooring, going to remove some img each epoch
        batch_count = math.floor(x_train.shape[0] / batch_size)

        if save_interval > 0 and noise_input == None:
            noise_input = np.random.uniform(-1.0, 1.0, size=[100, gen_input_dim])
        
        init_write_logs("gan")
        init_write_logs("discriminator")

        for e in range(1, epochs+1):
            indx_permutations = np.arange(x_train.shape[0])
            np.random.shuffle(indx_permutations)
            
            for i in range(batch_count):

                images_train = x_train[indx_permutations[i*batch_size:(i+1)*batch_size]]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                images_fake = generator.predict(noise)

                if model_type == "cnn":
                    images_train = np.expand_dims(np.array(images_train), axis=-1)

                x = np.concatenate((images_train, images_fake))
                y = np.ones([2*batch_size, 1])
                y[batch_size:, :] = 0
                discriminator.trainable = True
                d_loss = discriminator.train_on_batch(x, y)
                discriminator.trainable = False
                
                y = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                g_loss = gan.train_on_batch(noise, y)
                
                log_mesg = "\rEpoch %d - %d%%: [Discriminator loss: %f, acc: %f]" % (e, (i+1)/batch_count * 100 ,d_loss[0], d_loss[1])
                log_mesg = "%s [GAN loss: %f, acc: %f]" % (log_mesg, g_loss[0], g_loss[1])
                print(log_mesg, end="")

            # Append results of last batch of training
            append_line(d_loss[0], d_loss[1], "discriminator")
            append_line(g_loss[0], g_loss[1], "gan")

            if save_interval > 0:
                if (e % save_interval == 0) or e == epochs:
                    plot_generated_images(e, generator)

            print("\n  Validation...") # TODO

if model_type == "dense": 
    discriminator = dense_models.discriminator(disc_input_dim_flat)
    generator = dense_models.generator(gen_input_dim)
    gan = dense_models.gan(discriminator, generator, gen_input_dim)
    train(discriminator, generator, gan, epochs=epochs, save_interval=save_interval,
          batch_size=batch_size)

elif model_type == "cnn":
    ld = lambda : load_data(flatten=False)
    discriminator = cnn_models.discriminator(disc_input_dim)
    generator = cnn_models.generator(gen_input_dim)
    gan = cnn_models.gan(discriminator, generator, gen_input_dim)
    train(discriminator, generator, gan, epochs=epochs, save_interval=save_interval,
          batch_size=batch_size, load_data_func=ld)

else:
    print("Conditional GAN not implemented yet.")


