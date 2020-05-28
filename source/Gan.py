from tensorflow.keras.layers import Dense, Input, LeakyReLU, Flatten, Reshape, Dropout, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import random, ones_like, zeros_like, GradientTape, function
from matplotlib.transforms import Bbox
from tensorflow import io
import tensorflow as tf
import os
import time
import sys
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
sys.path.append("/")
from DataGenerator import DataGenerator


class Gan:
    def __init__(self, datasetPath, imgDims, batchSize, noiseDim):
        self.datasetPath = datasetPath

        self.dataGenerator = DataGenerator(imgDims, datasetPath)

        self.numFiles = len(os.listdir(datasetPath))
        self.batchSize = batchSize
        self.stepsPerEpoch = int(self.numFiles / self.batchSize)

        self.noiseDim = noiseDim
        self.imgDims = imgDims

        self.imageSavePath = '../generated_images'
        if not os.path.isdir(self.imageSavePath):
            os.mkdir(self.imageSavePath)
        self.modelSavePath = '../saved_models'
        if not os.path.isdir(self.modelSavePath):
            os.mkdir(self.modelSavePath)

        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.generatorOptimizer = Adam(1e-4)
        self.discriminatorOptimizer = Adam(1e-4)
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

    def create_generator(self):
        model = Sequential()
        model.add(Dense(int(self.imgDims[0] / 4) * int(self.imgDims[0] / 4) * 256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((int(self.imgDims[0] / 4), int(self.imgDims[0] / 4), 256)))
        assert model.output_shape == (None, int(self.imgDims[0] / 4), int(self.imgDims[0] / 4), 256)

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.imgDims[0] / 4), int(self.imgDims[1] / 4), 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.imgDims[0] / 2), int(self.imgDims[1] / 2), 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.imgDims[0], self.imgDims[1], self.imgDims[2])

        print("generator architecture")
        model.summary()

        return model

    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[self.imgDims[0], self.imgDims[1], self.imgDims[2]]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        print("discriminator architecture")
        model.summary()

        return model

    def saveModel(self, epoch):
        self.generator.save(os.path.join(self.modelSavePath, f"generator_at_epoch{epoch}.h5"))
        self.discriminator.save(os.path.join(self.modelSavePath, f"discriminator_at_epoch{epoch}.h5"))

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        for i in range(predictions.shape[0]):
            my_dpi = 100
            fig, ax = plt.subplots(1, figsize=(self.imgDims[0] / my_dpi, self.imgDims[0] / my_dpi), dpi=my_dpi)
            ax.set_position([0, 0, 1, 1])

            plt.imshow(np.asarray(predictions[i, :, :, 0] * 127.5 + 127.5, dtype='uint8'), cmap='gray')
            plt.axis('off')

            fig.savefig(os.path.join(self.imageSavePath, f'image_at_epoch_{epoch}_#{i}.png'),
                        bbox_inches=Bbox([[0, 0], [self.imgDims[0] / my_dpi, self.imgDims[0] / my_dpi]]),
                        dpi=my_dpi)
            plt.close()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(ones_like(fake_output), fake_output)

    @function
    def train_step(self, images):
        noise = random.normal([self.batchSize, self.noiseDim])

        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs, checkpointFrequency):
        num_examples_to_generate = 5

        seed = random.normal([num_examples_to_generate, self.noiseDim])

        for epoch in range(epochs):
            start = time.time()

            for batchNum in range(self.stepsPerEpoch):
                print(f"EPOCH = {epoch}; BATCH = {batchNum}/{self.stepsPerEpoch}")
                image_batch = self.dataGenerator.getBatch(self.batchSize)
                self.train_step(image_batch)

            if (epoch + 1) % checkpointFrequency == 0:
                self.generate_and_save_images(epoch + 1, seed)
                self.saveModel(epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        self.generate_and_save_images(epochs, seed)
