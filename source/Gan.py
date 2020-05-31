from tensorflow.keras.layers import Dense, Input, AveragePooling2D, UpSampling2D, LeakyReLU, Flatten, Reshape, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from matplotlib.transforms import Bbox
from tensorflow.keras import backend
from tensorflow.keras.constraints import max_norm
import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from CustomLayers import *


class Gan:
    def __init__(self, datasetPath, imgDims):
        """
        defines a progressively growing generative adversarial neural network
        :param datasetPath: path to the dataset containing grayscale .jpg images
        :param imgDims: target image dimensions (dataset-images will be resized)
        """
        self.datasetPath = datasetPath
        self.dataGenerator = DataGenerator(imgDims, datasetPath)
        self.imgDims = imgDims
        self.imageSavePath = '../generated_images'
        if not os.path.isdir(self.imageSavePath):
            os.mkdir(self.imageSavePath)
        self.modelSavePath = '../saved_models'
        if not os.path.isdir(self.modelSavePath):
            os.mkdir(self.modelSavePath)

        # number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
        self.n_blocks = 6
        # size of the latent space
        self.latent_dim = 100
        # define models
        self.d_models = self.define_discriminator(self.n_blocks)
        # define models
        self.g_models = self.define_generator(self.n_blocks)
        # define composite models
        self.gan_models = self.define_composite(self.d_models, self.g_models)
        self.n_batch = [16, 16, 16, 8, 4, 4]
        # 10 epochs == 500K images per training phase
        self.n_epochs = [5, 8, 8, 10, 10, 10]
        self.e_fadein = self.n_epochs
        self.e_norm = self.n_epochs

    def add_discriminator_block(self, old_model, n_input_layers=3):
        """
        adds a scaling stage to the discriminator architecture
        :param old_model: the model of the previous scaling stage
        :param n_input_layers: the number of input layers of the architecture as a whole
        :return:
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        # get shape of existing model
        in_shape = list(old_model.input.shape)
        # define new input shape as double the size
        input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
        in_image = Input(shape=input_shape)
        # define new input processing layer
        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # define new block
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        block_new = d
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model1 = Model(in_image, d)
        # compile model
        model1.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # downsample the new larger image
        downsample = AveragePooling2D()(in_image)
        # connect old input processing to downsampled new input
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        # fade in output of old model input layer with new input
        d = WeightedSum()([block_old, block_new])
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model2 = Model(in_image, d)
        # compile model
        model2.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        return [model1, model2]

    def define_discriminator(self, num_scale_stages, input_shape=(4, 4, 1)):
        """
        define the discriminator models for each image resolution
        :param num_scale_stages: the number of consecutive scaling stages
        :param input_shape: the initial input shape to begin with
        :return: a list containing a pair of discriminator-models for each scaling stage
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        model_list = list()
        # base model input
        in_image = Input(shape=input_shape)
        # conv 1x1
        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # conv 3x3 (output block)
        d = MinibatchStdev()(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # conv 4x4
        d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # dense output layer
        d = Flatten()(d)
        out_class = Dense(1)(d)
        # define model
        model = Model(in_image, out_class)
        # compile model
        model.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # store model
        model_list.append([model, model])
        # create submodels
        for i in range(1, num_scale_stages):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = self.add_discriminator_block(old_model)
            # store model
            model_list.append(models)
        return model_list

    # add a generator block
    def add_generator_block(self, old_model):
        """
        adds a generator scaling stage to the architecture
        :param old_model: the generator model of the previous scaling stage
        :return: a pair of generator models representing a new scaling stage
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        # get the end of the last block
        block_end = old_model.layers[-2].output
        # upsample, and define new block
        upsampling = UpSampling2D()(block_end)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # add new output layer
        out_image = Conv2D(1, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        # define model
        model1 = Model(old_model.input, out_image)
        # get the output layer from old model
        out_old = old_model.layers[-1]
        # connect the upsampling to the old output layer
        out_image2 = out_old(upsampling)
        # define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([out_image2, out_image])
        # define model
        model2 = Model(old_model.input, merged)

        return [model1, model2]

    # define generator models
    def define_generator(self, num_scale_stages, in_dim=4):
        """
        defines the generator architecture
        :param num_scale_stages: the number of consecutive scaling stages
        :param in_dim: the initial input dim to begin with
        :return: a list containing a pair of generator-models for each scaling stage
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        model_list = list()
        # base model latent input
        in_latent = Input(shape=(self.latent_dim,))
        # linear scale up to activation maps
        g = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
        g = Reshape((in_dim, in_dim, 128))(g)
        # conv 4x4, input block
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # conv 3x3
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # conv 1x1, output block
        out_image = Conv2D(1, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        # define model
        model = Model(in_latent, out_image)
        # store model
        model_list.append([model, model])
        # create submodels
        for i in range(1, num_scale_stages):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            old_model.summary()
            models = self.add_generator_block(old_model)
            # store model
            model_list.append(models)
        return model_list

    def define_composite(self, discriminators, generators):
        """
        define composite models for training generators via discriminators
        :param discriminators: a list of discriminator model-pairs for each scaling stage of the architecture
        :param generators: a list of generator model-pairs for each scaling stage of the architecture
        :return: a list containing the merged generator/discriminator scaling stages
        """
        model_list = list()
        # create composite models
        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]
            # straight-through model
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            # fade-in model
            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            # store
            model_list.append([model1, model2])
        return model_list

    def generate_real_samples(self, batchSize):
        """
        generates a batch from the dataset annotated with "1"
        :param batchSize: size of the batch
        :return: a numpy array representing a single batch, a vector representing the annotation
        """
        # select images
        X = self.dataGenerator.getBatch(batchSize)
        # generate class labels
        y = np.ones((batchSize, 1))
        return X, y

    def generate_latent_points(self, n_samples):
        """
        generates a latent input for the generator
        :param n_samples: length of the latent dimension
        :return: an input vector for the generator
        """
        # generate points in the latent space
        x_input = np.random.randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self, generator, batch_size):
        """
        use the generator to generate n fake examples, with class labels
        :param generator: the generator model
        :param batch_size: the batch size
        :return:
        """
        # generate points in latent space
        x_input = self.generate_latent_points(batch_size)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels
        y = -np.ones((batch_size, 1))
        return X, y

    def train_stage(self, g_model, d_model, gan_model, n_epochs, n_batch, fadein=False):
        """
        trains one scaling stage of the combined generator/discriminator architecture
        """
        # calculate the number of batches per training epoch
        bat_per_epo = int(self.dataGenerator.numFiles / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_steps):
            # update alpha for all WeightedSum layers when fading in new blocks
            if fadein:
                self.update_fadein([g_model, d_model, gan_model], i, n_steps)
            # prepare real and fake samples
            X_real, y_real = self.generate_real_samples(half_batch)
            X_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
            # update discriminator model
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # update the generator via the discriminator's error
            z_input = self.generate_latent_points(n_batch)
            y_real2 = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(z_input, y_real2)
            # summarize loss on this batch
            print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, d_loss1, d_loss2, g_loss))

    def save_results(self, status, g_model, n_samples=25):
        """
        saves the current state of the model as a .h5 file and a result plot as a .png file
        :param status: a string representing the type of result ("tuned", "faded")
        :param g_model: the generator model
        :param n_samples: the number of samples to generate as a plot
        :return:
        """
        # devise name
        gen_shape = g_model.output_shape
        name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
        # generate images
        X, _ = self.generate_fake_samples(g_model, n_samples)
        # normalize pixel values to the range [0,1]
        X = (X - X.min()) / (X.max() - X.min())
        # plot real images
        square = int(sqrt(n_samples))
        for i in range(n_samples):
            plt.subplot(square, square, 1 + i)
            plt.axis('off')
            plt.imshow(X[i][:, :, 0], cmap="Greys")
        # save plot to file
        plt.savefig(os.path.join(self.imageSavePath, f"plot_{name}.png"))
        plt.close()
        # save the generator model
        g_model.save(os.path.join(self.modelSavePath, f"generator_{name}.h5"))

    def train(self):
        """
        train the combined architecture
        """
        # fit the baseline model
        g_normal, d_normal, gan_normal = self.g_models[0][0], self.d_models[0][0], self.gan_models[0][0]
        # update the image shape
        gen_shape = g_normal.output_shape
        self.dataGenerator.currentImgShape = gen_shape[1:]
        # train normal or straight-through models
        self.train_stage(g_normal, d_normal, gan_normal, self.e_norm[0], self.n_batch[0])
        self.save_results('tuned', g_normal)
        # process each level of growth
        for i in range(1, len(self.g_models)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = self.g_models[i]
            [d_normal, d_fadein] = self.d_models[i]
            [gan_normal, gan_fadein] = self.gan_models[i]
            # scale dataset to appropriate size
            gen_shape = g_normal.output_shape
            self.dataGenerator.currentImgShape = gen_shape[1:]
            # train fade-in models for next level of growth
            self.train_stage(g_fadein, d_fadein, gan_fadein, self.e_fadein[i], self.n_batch[i], True)
            self.save_results('faded', g_fadein)
            # train normal or straight-through models
            self.train_stage(g_normal, d_normal, gan_normal, self.e_norm[i], self.n_batch[i])
            self.save_results('tuned', g_normal)

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

    @staticmethod
    def update_fadein(models, step, n_steps):
        """update the fade-in process of two models
        :param models: the models to fade between
        :param step: the current step within the fade
        :param n_steps: the total number of steps in the fade
        """
        # calculate current alpha (linear from 0 to 1)
        alpha = step / float(n_steps - 1)
        # update the alpha for each model
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    backend.set_value(layer.alpha, alpha)

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return backend.mean(y_true * y_pred)
