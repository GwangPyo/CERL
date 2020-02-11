'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
import os

from PIL import Image


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def get_dataset():
    ar = np.load("img_sample/sample_array.npy")
    np.random.shuffle(ar)

    return ar[100:], ar[:100]

# MNIST dataset
x_train, x_test = get_dataset()

image_size = x_train.shape[1]
width = x_train.shape[2]
x_train = np.reshape(x_train, [-1, image_size, width, 1])
x_test = np.reshape(x_test, [-1, image_size, width, 1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# network parameters
input_shape = (image_size, width, 1)
print(input_shape)
batch_size = 128
latent_dim = 42
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(filters=32, kernel_size=8, activation='relu', strides=4, padding='valid')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='valid')(x)
x = BatchNormalization()(x)
# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = BatchNormalization()(x)
x = Conv2DTranspose(filters=128, kernel_size=3, activation='relu', strides=1, padding='valid')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='valid')(x)
x = BatchNormalization()(x)
outputs = Conv2DTranspose(filters=1, kernel_size=8, strides=4, activation='sigmoid', padding='valid', name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    models = (encoder, decoder)
    data = x_test

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='Adam')
    vae.summary()
    # plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    #train the autoencoder

    x_train, x_test = get_dataset()
    image_size = x_train.shape[1]

    vae.load_weights('vae_model.h5')
    encoder.load_weights('vae_encoder.h5')
    decoder.load_weights('vae_decoder.h5')

    img = x_test[0]
    recon_img = vae.predict_on_batch(x_test)
    recon_img = recon_img[0]

    img = np.asarray(img * 255, dtype=np.uint8)

    img = Image.fromarray(np.squeeze(np.asarray(img, dtype=np.uint8)))
    img.show()
    recon_img = Image.fromarray(np.squeeze(np.asarray(recon_img * 255 , dtype=np.uint8)))
    recon_img.show()

    vae.fit(x_train,
            epochs=10,
            batch_size=batch_size,
            validation_data=(x_test, None))

    vae.save('vae_model.h5')
    encoder.save('vae_encoder.h5')
    decoder.save('vae_decoder.h5')
        # plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")