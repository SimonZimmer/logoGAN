from Gan import Gan

gan = Gan("../mnisttest",
          imgDims=(128, 128, 1),
          batchSize=64,
          noiseDim=100)

gan.train(epochs=10000,
          checkpointFrequency=1)
