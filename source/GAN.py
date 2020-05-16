from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time
from PIL import Image
import sys
sys.path.append("/")
import DataGenerator


np.random.seed(10)
noise_dim = 10
batch_size = 16
epochs = 30000

imageSavePath = '../fcgan-images'
modelSavePath = '../saved-models'
datasetPath = '../dataset/train/'
memmapPath = '../dataset/train.dat'
numFiles = len(os.listdir(datasetPath))
steps_per_epoch = int(numFiles / batch_size)

img_rows, img_cols, channels = 128, 128, 1
optimizer = Adam(0.0002, 0.5)

if not os.path.isdir(imageSavePath):
    os.mkdir(imageSavePath)
if not os.path.isdir(modelSavePath):
    os.mkdir(modelSavePath)


def create_generator():
    generator = Sequential()

    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(img_rows * img_cols * channels, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_discriminator():
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=img_rows * img_cols * channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def save_images(noise):
    generated_images = generator.predict(noise)
    for i, image in enumerate(generated_images):
        img = Image.fromarray(image.reshape((img_rows, img_cols)).astype(np.uint8))
        img.save(os.path.join(imageSavePath, f"generatedLogos_{time.time()}.jpg"), format='JPEG')


def save_model(generator, discriminator, epoch):
    generator.save(os.path.join(modelSavePath, f"epoch{epoch}_{time.time()}_generator.h5"))
    discriminator.save(os.path.join(modelSavePath, f"epoch{epoch}_{time.time()}_discriminator.h5"))


memmap = np.memmap(memmapPath, dtype='float32', mode='r+', shape=(img_rows, img_cols, numFiles))

discriminator = create_discriminator()
generator = create_generator()
discriminator.trainable = False

gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

for epochCount, epoch in enumerate(range(epochs)):
    if epochCount % 1 == 0:
        noise = np.random.normal(0, 1, size=(10, noise_dim))
        save_images(noise)
        save_model(generator, discriminator, epochCount)
    for n, batch in enumerate(range(steps_per_epoch)):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise)

        indices = np.random.randint(0, numFiles, size=batch_size)
        real_x = DataGenerator.getPreprocessedImg(memmap, indices)
        real_x = real_x.reshape(-1, img_rows*img_cols*channels)

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

        print(f'Epoch: {epoch} \t  Batch: {n} / {steps_per_epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
