from tensorflow.keras.layers import Dense, Input, AveragePooling2D, UpSampling2D, LeakyReLU, Flatten, Reshape, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from matplotlib.transforms import Bbox
from tensorflow.keras import backend
from tensorflow.keras.constraints import max_norm
import os
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
        self.dataGenerator = DataGenerator(imgDims, datasetPath)
        self.imgDims = imgDims
        self.imageSavePath = '../generated_images'
        if not os.path.isdir(self.imageSavePath):
            os.mkdir(self.imageSavePath)
        self.modelSavePath = '../saved_models'
        if not os.path.isdir(self.modelSavePath):
            os.mkdir(self.modelSavePath)

        self.num_scaling_stages = 6
        self.latent_dim = 100
        # define models
        self.g_models = self.define_generator(self.n_blocks)
        # define composite models
        self.gan_models = self.define_composite(self.d_models, self.g_models)
        self.n_batch = [256, 256, 256, 128, 64, 64]
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
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        # get shape of existing model
        old_input_shape = list(old_model.input.shape)
        # define new input shape as double the size
        new_input_shape = (old_input_shape[-2] * 2, old_input_shape[-2] * 2, old_input_shape[-1])

        inputLayer = Input(shape=new_input_shape)
        discriminator = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(inputLayer)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)

        discriminator = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = AveragePooling2D()(discriminator)
        new_scaling_stage = discriminator
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            discriminator = old_model.layers[i](discriminator)
        # define straight-through model
        model1 = Model(inputLayer, discriminator)
        # compile model
        model1.compile(loss=self.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # downsample the new larger image
        downsample = AveragePooling2D()(inputLayer)
        # connect old input processing to downsampled new input
        old_scaling_stage = old_model.layers[1](downsample)
        old_scaling_stage = old_model.layers[2](old_scaling_stage)
        # fade in output of old model input layer with new input
        discriminator = WeightedSum()([old_scaling_stage, new_scaling_stage])
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            discriminator = old_model.layers[i](discriminator)
        # define straight-through model
        model2 = Model(inputLayer, discriminator)
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
        discriminator = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        # conv 3x3 (output block)
        discriminator = MinibatchStdev()(discriminator)
        discriminator = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        # conv 4x4
        discriminator = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        # dense output layer
        discriminator = Flatten()(discriminator)
        out_class = Dense(1)(discriminator)
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

    @staticmethod
    def add_generator_block(old_model):
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
        batch = self.dataGenerator.getBatch(batchSize)
        labels = np.full((batchSize, 1), 0.9)
        plt.imshow(batch[1, :, :, 0])
        plt.savefig("test.png")
        return batch, labels

    def generate_latent_points(self, n_samples):
        """
        generates a latent input for the generator
        :param n_samples: length of the latent dimension
        :return: an input vector for the generator
        """
        latent_input = np.random.randn(self.latent_dim * n_samples)
        latent_input = latent_input.reshape(n_samples, self.latent_dim)
        return latent_input

    def generate_fake_samples(self, generator, batch_size):
        """
        use the generator to generate n fake examples, with class labels
        :param generator: the generator model
        :param batch_size: the batch size
        :return:
        """
        latent_input = self.generate_latent_points(batch_size)
        images = generator.predict(latent_input)
        labels = np.full((batch_size, 1), -0.9)
        return images, labels

    def train_stage(self, g_model, d_model, gan_model, n_epochs, batch_size, fadein=False):
        """
        trains one scaling stage of the combined generator/discriminator architecture
        """
        bat_per_epoch = int(self.dataGenerator.numFiles / batch_size)
        n_steps = bat_per_epoch * n_epochs
        half_batch = int(batch_size / 2)
        for i in range(n_steps):
            # update alpha for all WeightedSum layers when fading in new blocks
            if fadein:
                self.update_fadein([g_model, d_model, gan_model], i, n_steps)
            # prepare real and fake samples
            images_real, labels_real = self.generate_real_samples(half_batch)
            images_fake, labels_fake = self.generate_fake_samples(g_model, half_batch)
            # update discriminator model
            discriminator_loss1 = d_model.train_on_batch(images_real, labels_real)
            discriminator_loss2 = d_model.train_on_batch(images_fake, labels_fake)
            # update the generator via the discriminator's error
            latent_input = self.generate_latent_points(batch_size)
            labels_real2 = np.ones((batch_size, 1))
            generator_loss = gan_model.train_on_batch(latent_input, labels_real2)

            print(f'scaling stage {self.dataGenerator.currentImgShape[:2]}, '
                  f'step {i+1}/{n_steps}, discriminator1_loss={np.round(discriminator_loss1, 4)}, '
                  f'discriminator2_loss={np.round(discriminator_loss2, 4)}, '
                  f'generator_loss={np.round(generator_loss, 4)}')

    def save_results(self, status, g_model, n_samples=10):
        """
        saves the current state of the model as a .h5 file and a result plot as a .png file
        :param status: a string representing the type of result ("tuned", "faded")
        :param g_model: the generator model
        :param n_samples: the number of samples to generate as a plot
        :return:
        """
        gen_shape = g_model.output_shape
        name = f'{gen_shape[1]}x{gen_shape[2]}-{status}'
        images, _ = self.generate_fake_samples(g_model, n_samples)
        images = self.normalize_images(images)
        for i in range(n_samples):
            my_dpi = 100
            fig, ax = plt.subplots(1, figsize=(self.imgDims[0] / my_dpi, self.imgDims[0] / my_dpi), dpi=my_dpi)
            ax.set_position([0, 0, 1, 1])

            plt.imshow(images[i][:, :, 0], cmap="Greys")
            plt.axis('off')

            fig.savefig(os.path.join(self.imageSavePath, f"plot_{name}_#{i}.png"),
                        bbox_inches=Bbox([[0, 0], [self.imgDims[0] / my_dpi, self.imgDims[0] / my_dpi]]), dpi=my_dpi)
            plt.close()

        g_model.save(os.path.join(self.modelSavePath, f"generator_{name}.h5"))

    def train(self):
        """
        train the combined architecture
        """
        # fit the baseline model
        generator_normal, discriminator_normal, gan_normal = self.generator_architectures[0][0], self.discriminator_architectures[0][0], self.gan_models[0][0]
        # update the image shape
        gen_shape = generator_normal.output_shape
        self.dataGenerator.currentImgShape = gen_shape[1:]
        # train normal or straight-through models
        self.train_stage(generator_normal, discriminator_normal, gan_normal, self.e_norm[0], self.batch_sizes[0])
        self.save_results('tuned', generator_normal)
        # process each level of growth
        for i in range(1, len(self.generator_architectures)):
            # retrieve models for this level of growth
            [generator_normal, generator_fadein] = self.generator_architectures[i]
            [discriminator_normal, discriminator_fadein] = self.discriminator_architectures[i]
            [gan_normal, gan_fadein] = self.gan_models[i]
            # scale dataset to appropriate size
            gen_shape = generator_normal.output_shape
            self.dataGenerator.currentImgShape = gen_shape[1:]
            # train fade-in models for next level of growth
            self.train_stage(generator_fadein, discriminator_fadein, gan_fadein, self.e_fadein[i], self.batch_sizes[i], True)
            self.save_results('faded', generator_fadein)
            # train normal or straight-through models
            self.train_stage(generator_normal, discriminator_normal, gan_normal, self.e_norm[i], self.batch_sizes[i])
            self.save_results('tuned', generator_normal)

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

    @staticmethod
    def normalize_images(images):
        return (images - images.min()) / (images.max() - images.min())
