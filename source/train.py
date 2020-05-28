from Gan import Gan

gan = Gan("../dataset/train",
          imgDims=(128, 128, 1),
          batchSize=128,
          noiseDim=100)

gan.train(epochs=10000,
          checkpointFrequency=50)

