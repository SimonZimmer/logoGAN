from Gan import Gan

gan = Gan("/Users/simonzimmermann/Downloads/mnistasjpg",
          imgDims=(28, 28, 1),
          batchSize=128,
          noiseDim=100)

gan.train(epochs=10000,
          checkpointFrequency=50)
