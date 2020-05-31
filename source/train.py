from Gan import Gan

gan = Gan("../mnisttest",
          imgDims=(128, 128, 1))

gan.train()
