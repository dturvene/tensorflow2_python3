#!/usr/bin/env python3
"""
unit test for packages and keras support

* https://www.tensorflow.org/tutorials/keras/regression
"""

import sys
from pdb import set_trace as bp
import unittest

import timeit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)

def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)
        
def ut_cpu_gpu():
    '''
    shamelessly culled from
    https://colab.research.google.com/notebooks/gpu.ipynb
    '''
    devname = tf.test.gpu_device_name()
    if devname != '/device:GPU:0':
        print(f'\n\ndevice={devname} shows a GPU is not present.\n'
              'See if nvidia-uvm driver is loaded in host and then\n'
              'reload this unit test')
        return(0)

    # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
    # this is only necessary for the first time a GPU is accessed
    cpu()
    print('warming up gpu')
    gpu()

    # Run CPU N times then GPU and compare runtime ratio
    print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
          '(batch x height x width x channel). Sum of ten runs.')
    print('CPU (s):')
    cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
    print(cpu_time)
    print('GPU (s):')
    gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
    print(gpu_time)
    print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))    

class Ut(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    #@unittest.skip('good')
    def test1(self):
        tf.test.is_gpu_available()
    #@unittest.skip('good')
    def test2(self):
        print(tf.reduce_sum(tf.random.normal([1000, 1000])))
    def test3(self):
        ut_cpu_gpu()        

if __name__ == '__main__':
    # exec(open('./ut_gpu.py').read())
    print('tf={}'.format(tf.__version__))
    unittest.main(exit=False)
