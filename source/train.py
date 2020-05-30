from Gan import Gan

gan = Gan("../dataset/train",
          imgDims=(28, 28, 1),
          batchSize=64,
          noiseDim=100)

gan.train(epochs=10000,
          checkpointFrequency=100)

