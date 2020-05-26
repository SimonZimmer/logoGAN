from Gan import Gan

gan = Gan("../dataset",
          imgDims=(28, 28, 1),
          batchSize=256,
          noiseDim=100)

gan.train(epochs=10000,
          checkpointFrequency=1)
