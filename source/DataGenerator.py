import numpy as np
import os
from PIL import Image


def getMinDimSize(filePath):
    allMinSizes = []
    minImgSize = 28
    for imgFile in os.listdir(filePath):
        imgFilePath = os.path.join(filePath, imgFile)
        if imgFile.endswith(".jpg"):
            img = Image.open(imgFilePath)
            minSize = np.min(img.size)
            if minSize < minImgSize:
                os.remove(imgFilePath)
            allMinSizes.append(minSize)
        else:
            os.remove(imgFilePath)
    globalMinSize = np.min(allMinSizes)

    return globalMinSize


def createMemmap(datasetPath, memmapPath, imgDims):
    allFiles = os.listdir(datasetPath)
    numFiles = len(allFiles)
    memmap = np.memmap(memmapPath, dtype='float32', mode='w+', shape=(*imgDims, numFiles))
    for n, imgFile in enumerate(allFiles):
        if int(n == numFiles / 2):
            del memmap
            memmap = np.memmap(memmapPath, dtype='float32', mode='w+', shape=(*imgDims, numFiles))
        print(f'Writing {n}/{numFiles} ({imgFile})')
        imgFilePath = os.path.join(datasetPath, imgFile)
        img = Image.open(imgFilePath)
        try:
            imgBw = img.convert('L')
            imgBwResized = imgBw.resize((128, 128), Image.NEAREST)
            imgData = np.array(imgBwResized)
            memmap[:, :, n] = imgData
        except OSError:
            os.remove(imgFilePath)
    print('done.')

    return (*imgDims, numFiles)


def getPreprocessedImg(memmap, indices):
    imgArrays = []
    for n, i in enumerate(indices):
        img = memmap[:, :, i]
        imgArrays.append(img)
    batch = np.stack(imgArrays, axis=2)

    return batch
