import math;
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras.backend as K
import os.path

import models.dense_models as dense_models
import models.cnn_models as cnn_models
import models.conditional_modals as cond_models

save_path = "./imgs/"
log_dir = "./logs/"
disc_input_dim_flat = 784
disc_input_dim = (28,28,1)
gen_input_dim = 100
model_type = "conditional" # cnn / dense / conditional

save_interval = 5
epochs = 100
batch_size = 256

def load_data(flatten=True):
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    # Flatten for the dense network, otherwise expand dimensions for cond and cnn
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    else:
        x_train = np.expand_dims(np.array(x_train), axis=-1)

    return (x_train, y_train, x_test, y_test)


def plot_generated_images(epoch, generator, noise, labels=None, examples=100, dim=(10,10),
                          figsize=(10,10), model_type=model_type, classes=10):
    # TODO: Refactor, conditonal made everything a little different

    if labels is None:
        generated_images = generator.predict(noise) 
    else:
        generated_images = generator.predict([noise, labels])

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

def check_overwrite_ok(model_type=model_type):
    path = "{0}{1}_{2}.txt".format(log_dir, model_type, "gan")
    if os.path.exists(path):
        key = input("Data for \"{0}\" already exist. You are about to overwrite it. Press \"y\" to continue: ".format(model_type))
        if key == "y":
            print("Overwriting files.")
            return True
        else:
            print("Aborting.")
            return False
    return True

def train(discriminator, generator, gan, load_data_func=load_data, 
          gen_input_dim=100, epochs=400, batch_size=256, classes=10, save_interval=0):

        if not check_overwrite_ok():
            return

        (x_train, y_train, _, _) = load_data_func()

        # Due to flooring, going to remove some img each epoch
        batch_count = math.floor(x_train.shape[0] / batch_size)

        if save_interval > 0:
            noise_plot = np.random.uniform(-1.0, 1.0, size=[100, gen_input_dim])
            labels_plot = None
            if model_type == "conditional":
                labels_plot = np.random.randint(0, classes, 100)

        init_write_logs("gan")
        init_write_logs("discriminator")

        for e in range(1, epochs+1):
            indx_permutations = np.arange(x_train.shape[0])
            np.random.shuffle(indx_permutations)
            
            for i in range(batch_count):

                images_train = x_train[indx_permutations[i*batch_size:(i+1)*batch_size]]

                if not model_type == "conditional": # training of Dense and CNN model

                    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                    images_fake = generator.predict(noise)
                    x = np.concatenate((images_train, images_fake))

                    y = np.ones([2*batch_size, 1])
                    y[batch_size:, :] = 0

                    discriminator.trainable = True
                    d_loss = discriminator.train_on_batch(x, y)
                    discriminator.trainable = False
                    
                    y = np.ones([batch_size, 1])
                    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                    g_loss = gan.train_on_batch(noise, y)

                else: # Train conditional modal
                    
                    # Create fake images with generator
                    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                    labels = np.random.randint(0, classes, batch_size)
                    images_fake = generator.predict([noise, labels])

                    # prepare data and train discriminator on fake and real images
                    x = np.concatenate((images_train, images_fake))
                    y = np.ones([2*batch_size, 1])
                    y[batch_size:, :] = 0
                    # TODO: check whether splitting up necessary, i.e. first real then fake
                    labels_train = y_train[indx_permutations[i*batch_size:(i+1)*batch_size]]
                    labels_fake = np.random.randint(0, 10, batch_size)
                    labels = np.concatenate([labels_train, labels_fake])
                    discriminator.trainable = True
                    d_loss = discriminator.train_on_batch([x, labels], y)
                    discriminator.trainable = False

                    # Train GAN
                    y = np.ones([batch_size, 1])
                    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, gen_input_dim])
                    labels = np.random.randint(0, classes, batch_size)
                    g_loss = gan.train_on_batch([noise, labels], y)

                log_mesg = "\rEpoch %d - %d%%: [Discriminator loss: %f, acc: %f]" % (e, (i+1)/batch_count * 100 ,d_loss[0], d_loss[1])
                log_mesg = "%s [GAN loss: %f, acc: %f]" % (log_mesg, g_loss[0], g_loss[1])
                print(log_mesg, end="")

            # Append results of last batch of training
            append_line(d_loss[0], d_loss[1], "discriminator")
            append_line(g_loss[0], g_loss[1], "gan")

            if save_interval > 0:
                if (e % save_interval == 0) or e == epochs:
                    plot_generated_images(e, generator, noise_plot, labels_plot)

            print("")

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

elif model_type == "conditional":
    ld = lambda : load_data(flatten=False)
    discriminator = cond_models.discriminator(disc_input_dim)
    generator = cond_models.generator(gen_input_dim)
    gan = cond_models.gan(discriminator, generator)
    train(discriminator, generator, gan, epochs=epochs, save_interval=save_interval,
          batch_size=batch_size, load_data_func=ld)

else:
    print("Incompatible mode type")
