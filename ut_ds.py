#!/usr/bin/env python3
"""
unit tests for dataset loading

* https://www.tensorflow.org/datasets/catalog/overview
tensoflow catalog of 122 ready-to-use datasets 

These use the `tensorflow_datasets` API installed with
`pip install tensorflow-datasets`

* https://www.tensorflow.org/datasets/api_docs/python/tfds
TFDS API: load data into a tf.data.Dataset instance

* https://www.tensorflow.org/api_docs/python/tf/data/Dataset
tf.data.Dataset API: shuffle, batch, prefetch
"""

import os
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

# https://vis-www.cs.umass.edu/lfw/
def ut_lfw():
    '''
    lfw: https://www.tensorflow.org/datasets/catalog/lfw
    Labeled Faces in the Wild: a DB for studying face recognition in Unconstrained Environments
    TRAIN: 13,233
    '''

    print('THIS TAKES A LONG TIME FOR INITIAL DOWNLOAD to /home/user1/tensorflow_datasets!')

    ds_tra = tfds.load(name='lfw', split=tfds.Split.TRAIN) # 13233
    # dataset input pipeline:
    # - prefetch a dynamic set
    # - combine consecutive elements into a batch N%16 (828)
    # - shuffle first 1024 elements in the batch
    ds = ds_tra.shuffle(1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    len(ds) # 828 batches of 16 elems (13233/16)
    # each dictionary entry is a 
    
    if True:
        # ds is an iterator so use list comprehension to read all batchs into a list
        ebatchs = [item for item in ds]
        len(ebatchs) # 828 batchs of 16 elements
        # pull a random batch from list
        batch1 = ebatchs[73]
    else:
        # OR pull a single batch (seems to be random) into a list and get the batch
        batch1 = [item for item in ds.take(1)][0]

    type(batch1)
    # a dictionary of two fields:
    #   'image': a tensor of dtype=uint8
    #      shape = (16, 250, 250, 3) or 16 images of 250x250x3(RGB 8b)
    #   'label': a tensor of dtype=string
    #      shape (16, ) or 16 strings, each a label for the image
    #         
    batch1.keys()

    # inspect a tensor
    timage = batch1['image']
    type(timage)
    timage.ndim

    shape = timage.get_shape()
    type(shape)
    shape.dims

    # get fields
    print('elem 2 label tensor:\n', batch1['label'][2])
    print('elem 2 image tensor:\n', batch1['image'][2])

    # convert tensor images to a numpy array
    bnumpy = batch1['image'].numpy()
    print('elem 2 image numpy.ndarray:\n', bnumpy[2])
    print(f'elem 2 row 0:\nsize={len(bnumpy[2][0])}\narray=\n{bnumpy[2][0]}')
    
    # imgs=d1[0]['image']  # 16 images in batch
    # img=imgs.numpy()[2]  # get second image as numpy.ndarray, 250x250x3(RGB 8b)
    
def ut_lfw_show():
    '''
    show all LFW images in a single batch (16)
    using matplotlib
    '''

    ds = tfds.load(name='lfw', split=tfds.Split.TRAIN) # 13233
    ds = ds.shuffle(1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    fig = plt.figure(figsize=(16,16))

    # 4D conv2D layer (batch, rows, cols, channels)
    for features in ds.take(1):
        img4d = features['image']
        print('i={}, label={}'.format(img4d.shape, features['label']))
        for i in range(img4d.shape[0]):
            a = fig.add_subplot(4,4,i+1)
            plt.imshow(img4d[i])
        plt.show()

def ut_builders():
    '''
    https://www.tensorflow.org/datasets/overview
    '''
    builders = tfds.list_builders()
    # list of valid dataset names

    # find the one I want
    dsname = [rec for rec in builders if rec == 'mnist']

    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)
    assert isinstance(ds_train, tf.data.Dataset) and isinstance(ds_test, tf.data.Dataset) 
  

def ut_titanic():
    '''
    Tensorflow Dataset overview
    https://www.tensorflow.org/datasets/overview

    Dataset catalog
    https://www.tensorflow.org/datasets/catalog/overview

    OpenML Titanic Ds
    https://www.openml.org/search?type=data&sort=runs&id=40945&status=active
    '''

    # tfds.list_builders()
    # ds = tfds.load('titanic', split='train', shuffle_files=True)

    ds = tfds.load('titanic', split='train', shuffle_files=False)
    # python prompt shows:
    #   Dataset titanic downloaded and prepared to /home/user1/tensorflow_datasets/titanic/4.0.0.
    #   Subsequent calls will reuse this data.
    # ~/tensorflow_datasets/titanic/4.0.0/dataset_info.json: openml info, module, splits,
    #    list of features as dict, target
    # ~/tensorflow_datasets/titanic/4.0.0/titanic-train.tfrecord-00000-of-00001: binary formatted
    #    data for passengers (1309 records)
    
    print(f'titanic = {len(ds)}')

    for rec in ds:
        print(rec['name'])

    # ds = ds.take(1).cache().repeat()
    ds1 = ds.take(3)
    for rec in ds1:
        print(list(rec.keys()))
        # print(rec['age'], rec['name'])

    bp()

def ut_collections():
    '''
    https://www.tensorflow.org/datasets/dataset_collections

    '''
    # all_datasets = collection_loader.load_all_datasets()
    ds_collections = tfds.list_dataset_collections()

    collection_loader = tfds.dataset_collection('xtreme')

    # collection_loader.print_info()
    collection_loader.print_datasets()
    

    bp()
    
class Ut(unittest.TestCase):
    def setUp(self):
        os.environ['PAGER'] = 'cat'
    def tearDown(self):
        pass
    @unittest.skip('good')
    def test1(self):
        ut_lfw()
    # @unittest.skip('display only')        
    def test1_1(self):
        ut_lfw_show()
    @unittest.skip('work in progress')
    def test2(self):
        ut_titanic()
    @unittest.skip('work in progress')
    def test3(self):
        ut_collections()
        
  
if __name__ == '__main__':
    # exec(open('./ut_ds.py').read())
    print('tf={}'.format(tf.__version__))
    unittest.main(exit=False)
