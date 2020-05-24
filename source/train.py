from Gan import Gan

gan = Gan("/home/simon/datasets/mnist",
          imgDims=(28, 28, 1),
          batchSize=64,
          noiseDim=100)

gan.train(epochs=100,
          checkpointFrequency=1)
