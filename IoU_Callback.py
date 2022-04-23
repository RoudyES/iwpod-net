import tensorflow as tf
import statistics
import numpy as np
from shapely.geometry import Polygon
import os
import cv2
from src.keras_utils import detect_lp_width
from src.utils	import im2single

class IoUCallback(tf.keras.callbacks.Callback):
    """Calculates IoU of train and validation data each x number of epochs.

  Arguments:
      frequency: specifies how many training epochs to run before the IoU is calculated.
  """

    def __init__(self, trainDir, valDir= None, frequency=1):
        super(IoUCallback, self).__init__()
        self.frequency = frequency
        self.trainDir = trainDir
        self.valDir = valDir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            trainIous = []
            valIous = []
            for root, dirs, files in os.walk(self.trainDir):
                for filename in files:
                    fnamewoext = filename.split('.')[0]
                    ext = filename.split('.')[1]
                    pts = None
                    labelPoly = None
                    predictionPoly = None
                    if ext != 'txt':
                        img = cv2.imread(root + '/' + filename)
                        iwh = np.array(img.shape[1::-1],dtype=float).reshape((2,1))
                        Llp, LlpImgs,_ = detect_lp_width(self.model, im2single(img), 480, 2**4, tuple([240,8]), 0.35)
                        for i, img in enumerate(LlpImgs):
                            pts = Llp[i].pts * iwh
                            predictionPoly = Polygon([(pts[0,0], pts[1,0]), (pts[0,1], pts[1,1]), (pts[0,2], pts[1,2]), (pts[0,3],pts[1,3])])
                        with open(root + '/' + fnamewoext+'.txt') as f:
                            line = f.readlines()[0]
                            label = np.array(line.split(','))[1:9]
                            label = [float(x) for x in label]
                            label = np.array(label)
                            label[:4] = label[:4] * img.shape[1]
                            label[4:] = label[4:] * img.shape[0]
                            labelPoly = Polygon([(label[0], label[4]), (label[1], label[5]), (label[2], label[6]), (label[3],label[7])])
                        intersect = predictionPoly.intersection(labelPoly).area
                        union = predictionPoly.union(labelPoly).area
                        iou = intersect / union
                        trainIous.append(iou)
            print('Mean IOU of train set: ' + str(statistics.fmean(trainIous)))
            if self.valDir is not None:
                for root, dirs, files in os.walk(self.valDir):
                    for filename in files:
                        fnamewoext = filename.split('.')[0]
                        ext = filename.split('.')[1]
                        pts = None
                        labelPoly = None
                        predictionPoly = None
                        if ext != 'txt':
                            img = cv2.imread(root + '/' + filename)
                            iwh = np.array(img.shape[1::-1],dtype=float).reshape((2,1))
                            Llp, LlpImgs,_ = detect_lp_width(self.model, im2single(img), 480, 2**4, tuple([240,8]), 0.35)
                            for i, img in enumerate(LlpImgs):
                                pts = Llp[i].pts * iwh
                                predictionPoly = Polygon([(pts[0,0], pts[1,0]), (pts[0,1], pts[1,1]), (pts[0,2], pts[1,2]), (pts[0,3],pts[1,3])])
                            with open(root + '/' + fnamewoext+'.txt') as f:
                                line = f.readlines()[0]
                                label = np.array(line.split(','))[1:9]
                                label = [float(x) for x in label]
                                label = np.array(label)
                                label[:4] = label[:4] * img.shape[1]
                                label[4:] = label[4:] * img.shape[0]
                                labelPoly = Polygon([(label[0], label[4]), (label[1], label[5]), (label[2], label[6]), (label[3],label[7])])
                            intersect = predictionPoly.intersection(labelPoly).area
                            union = predictionPoly.union(labelPoly).area
                            iou = intersect / union
                            valIous.append(iou)
            print('Mean IOU of val set: ' + str(statistics.fmean(valIous)))

