from Gan import Gan

gan = Gan("../dataset/train",
          imgDims=(128, 128, 1))

gan.train()
