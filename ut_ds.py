#!/usr/bin/env python3
"""
unit tests for dataset loading
"""

import sys
from pdb import set_trace as bp
import unittest

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def ut_tfds():
    '''
    https://www.tensorflow.org/datasets
    
    https://www.tensorflow.org/datasets/catalog/overview
    122 datasets

    lfw: https://www.tensorflow.org/datasets/catalog/lfw 
    TRAIN: 13,233
    '''
    # print(tfds.list_builders())

    ds_tra = tfds.load(name='lfw', split=tfds.Split.TRAIN)
    ds = ds_tra.shuffle(1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    fig = plt.figure(figsize=(16,16))

    # 4D conv2D layer (batch, rows, cols, channels)
    for features in ds.take(1):
        img4d = features['image']
        print('i={}, label={}'.format(img4d.shape, features['label']))
        for i in range(img4d.shape[0]):
            a = fig.add_subplot(4,4,i+1)
            plt.imshow(img4d[i])
        plt.show()

class Ut(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    #@unittest.skip('good')
    def test1(self):
        ut_tfds()
  
if __name__ == '__main__':
    # exec(open('./ut_ds.py').read())
    print('tf={}'.format(tf.__version__))
    #unittest.main(exit=False)
