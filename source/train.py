from Gan import Gan

gan = Gan("../mnisttest",
          imgDims=(512, 512, 1))

gan.train()
