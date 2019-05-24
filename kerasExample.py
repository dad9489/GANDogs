from __future__ import print_function, division

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

model_path = 'images/dog_gen/models/'


class GAN:
    def __init__(self):
        self.img_rows = 150
        self.img_cols = 150
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.test_rows = 5
        self.test_cols = 5

        optimizer = Adam(0.0002, 0.5)

        if not load:
            self.test_noise = np.random.normal(0, 1, (self.test_rows * self.test_cols, self.latent_dim))
            with open(model_path + 'test_noise.dat', 'wb') as f:
                pickle.dump(self.test_noise, f)
        else:
            with open(model_path+'test_noise.dat', 'rb') as f:
                self.test_noise = pickle.load(f)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy']
                                   )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates images
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, starting_num=0, data_size=20569,
              model_weights_paths=None):

        # Load the dataset
        path = 'Images/cropped'
        data = ImageDataGenerator().flow_from_directory(path, target_size=(150, 150), batch_size=data_size)
        x_train = next(data)[0]

        # Rescale -1 to 1
        x_train = x_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        initialized = False if load else True

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if initialized:
                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_images(epoch)
            else:
                with open(optimizer_path, 'rb') as f:
                    self.discriminator.optimizer.set_weights(pickle.load(f))
                self.generator.load_weights(model_weights_paths[0])
                self.discriminator.load_weights(model_weights_paths[1])
                self.combined.load_weights(model_weights_paths[2])
                initialized = True

    def sample_images(self, epoch):
        r, c = self.test_rows, self.test_cols
        gen_imgs = self.generator.predict(self.test_noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.round(gen_imgs, 5)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :,:])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/dog_gen/%d.png" % epoch)

        save_freq = 300
        if epoch % save_freq == 0:
            if epoch != 0:
                prev = epoch - save_freq
                try:
                    os.remove(model_path + "generator_%d.h5" % prev)
                    os.remove(model_path + "discriminator_%d.h5" % prev)
                    os.remove(model_path + "combined_%d.h5" % prev)
                    os.remove(model_path + "optimizer_%d.pkl" % prev)
                except FileNotFoundError:
                    pass

            self.generator.save_weights(model_path + "generator_%d.h5" % epoch)
            self.discriminator.save_weights(model_path + "discriminator_%d.h5" % epoch)
            self.combined.save_weights(model_path + "combined_%d.h5" % epoch)
            with open(model_path+"optimizer_%d.pkl" % epoch, 'wb') as f:
                pickle.dump(self.discriminator.optimizer.get_weights(), f)

        plt.close()


if __name__ == '__main__':

    load = False
    load_epoch_num = 600

    if load:
        exten = '.h5'
        combined_path = model_path+'combined_'+str(load_epoch_num)+exten
        discriminator_path = model_path+'discriminator_'+str(load_epoch_num)+exten
        generator_path = model_path+'generator_'+str(load_epoch_num)+exten
        optimizer_path = model_path+'optimizer_'+str(load_epoch_num)+'.pkl'
        gan = GAN()
        gan.train(epochs=1000000, batch_size=32, sample_interval=50, starting_num=load_epoch_num,
                  model_weights_paths=(generator_path, discriminator_path, combined_path))
    else:
        gan = GAN()
        gan.train(epochs=1000000, batch_size=32, sample_interval=50)
