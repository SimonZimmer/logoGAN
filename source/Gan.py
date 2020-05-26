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
        model = Sequential()
        model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[28, 28, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        return model

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

        #TODO: Add noise to the real and generated images before feeding them into the discriminator.

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs, checkpointFrequency):
        num_examples_to_generate = 3

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

