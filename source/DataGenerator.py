import numpy as np
import os
from PIL import Image
from tensorflow import float32, convert_to_tensor
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, imgDims, datasetPath):
        self.removeNonImageFiles(datasetPath)
        self.imgDims = imgDims
        self.datasetPath = datasetPath
        self.memmapPath = os.path.join(self.datasetPath, 'train.dat')
        self.numFiles = len(os.listdir(datasetPath))
        if not os.path.isfile(self.memmapPath):
            self.createMemmap()

    def createMemmap(self):
        allFiles = os.listdir(self.datasetPath)
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='w+', shape=(*self.imgDims, self.numFiles))
        for n, imgFile in enumerate(allFiles):
            print(f'Writing {n}/{self.numFiles} ({imgFile})')
            imgFilePath = os.path.join(self.datasetPath, imgFile)
            memmap[:, :, :, n] = self.getProcessedImage(imgFilePath)
        del memmap
        print('done.')

    def getBatch(self, batchSize):
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='r', shape=(*self.imgDims, self.numFiles))
        indices = np.random.randint(0, self.numFiles-1, size=batchSize)
        imgArrays = []
        for i in indices:
            img = memmap[:, :, :, i]
            imgArrays.append(img)
        batch = np.stack(imgArrays, axis=0)
        batch = convert_to_tensor(batch, dtype=float32)
        del memmap

        return batch

    def getProcessedImage(self, imgFilePath):
        img = Image.open(imgFilePath)
        img = img.convert('L')
        img = img.resize(self.imgDims[:2], Image.LANCZOS)
        imgData = np.array(img)
        imgData = np.expand_dims(imgData, 2)
        imgData = np.subtract(np.divide(imgData, (255 * 0.5)), 1)

        return imgData

    @staticmethod
    def filterImgSize(filePath, imgSize=28):
        for imgFile in os.listdir(filePath):
            imgFilePath = os.path.join(filePath, imgFile)
            if imgFile.endswith(".jpg"):
                img = Image.open(imgFilePath)
                minSize = np.min(img.size)
                if minSize < imgSize:
                    os.remove(imgFilePath)

    @staticmethod
    def removeNonImageFiles(path):
        allFiles = os.listdir(path)
        for f in allFiles:
            print(f"scanning file {f}")
            if not f.endswith(".jpg") and not f.endswith(".jpeg"):
                os.remove(os.path.join(path, f))
