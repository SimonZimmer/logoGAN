from tensorflow.keras.layers import Dense, Input, LeakyReLU, Flatten, Reshape, Dropout, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import random, ones_like, zeros_like, GradientTape, function, enable_eager_execution
import os
import time
import sys
import PIL.Image as Image
sys.path.append("/")
from DataGenerator import DataGenerator


class Gan:
    def __init__(self, datasetPath, imgDims, batchSize, noiseDim):
        enable_eager_execution()
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
        generator = Sequential()
        internalDims = (int(self.imgDims[0] / 4), int(self.imgDims[1] / 4))

        generator.add(Dense(internalDims[0] * internalDims[1] * 256, use_bias=False, input_shape=(self.noiseDim,)))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        generator.add(Reshape((internalDims[0], internalDims[1], 256)))
        assert generator.output_shape == (None, internalDims[0], internalDims[1], 256)

        generator.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert generator.output_shape == (None, internalDims[0], internalDims[1], 128)
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        generator.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert generator.output_shape == (None, 2 * internalDims[0], 2 * internalDims[1], 64)
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        generator.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert generator.output_shape == (None, self.imgDims[0], self.imgDims[1], self.imgDims[2])

        return generator

    def create_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[*self.imgDims]))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.3))

        discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.3))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))

        return discriminator

    def saveModel(self, epoch):
        self.generator.save(os.path.join(self.modelSavePath, f"generator_at_epoch{epoch}.h5"))
        self.discriminator.save(os.path.join(self.modelSavePath, f"generator_at_epoch{epoch}.h5"))

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        for i in range(predictions.shape[0]):
            image = predictions[i, :, :, 0]

            imageData = image.numpy()
            imageData = imageData * 127.5 + 127.5
            array_buffer = imageData.tobytes()
            img = Image.new("I", imageData.T.shape)
            img.frombytes(array_buffer, 'raw', "I;16")
            img.save(os.path.join(self.imageSavePath, f"image_at_epoch_{epoch}_{i}.png"))

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
        num_examples_to_generate = 16

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

