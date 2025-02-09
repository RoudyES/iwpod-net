import enum
import numpy as np
import cv2

from tensorflow import keras
from src.sampler import augment_sample, labels2output_map
#
#  Creates ALPR Data Generator
#


class ALPRDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, batch_size=32, dim=208, stride= 16, shuffle=True, OutputScale=1.0):
        'Initialization'
        self.dim = dim
        self.stride = stride
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.OutputScale = OutputScale
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch. Pads training data to be a multiple of batch size'
#        self.indexes = list(np.arange(0, len(self.data), 1))
        self.indexes = list(np.arange(0, len(self.data), 1))
        self.indexes += list(np.random.choice(self.indexes, self.batch_size - len(self.data) % self.batch_size))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    #def __data_generation(self, indexes):
    #    # X : (n_samples, *dim, n_channels)
    #    'Generates data containing batch_size samples'
#
    #    X = np.empty((self.batch_size, self.dim, self.dim, 3))
    #    fpnOutput = [self.dim//res for res in self.strides]
    #    finalDim = fpnOutput[0]**2 + fpnOutput[1]**2 + fpnOutput[2]**2
    #    y = []
    #    #y = y.reshape((self.batch_size,-1,-1,9))
    #    #y = np.empty((self.batch_size, self.dim//self.stride, self.dim//self.stride, 9))
    #    # Generate data
    #    for i, stride in enumerate(self.strides):
    #        YY = np.empty((self.batch_size,fpnOutput[i],fpnOutput[i],9))
    #        for j, idx in enumerate(indexes):
    #            
    #            XX, llp, ptslist = augment_sample(self.data[idx][0], self.data[idx][1], self.dim)
    #            temp = labels2output_map(llp, ptslist, self.dim, stride, alfa=0.5)
    #            YY[j, ] = temp
    #            X[j, ] = XX*self.OutputScale
#
    #        y.append(YY)
    #    return X, [*y]

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        X = np.empty((self.batch_size, self.dim, self.dim, 3))
        y = np.empty((self.batch_size, self.dim//self.stride, self.dim//self.stride, 9))
        # Generate data
        for i, idx in enumerate(indexes):
            I = cv2.imread(self.data[idx][0])
            # Store sample
            XX, llp, ptslist = augment_sample(I, self.data[idx][1], self.dim)
            YY = labels2output_map(llp, ptslist, self.dim, self.stride, alfa = 0.5)
            X[i,] = XX*self.OutputScale
            y[i,] = YY
        return X, y
