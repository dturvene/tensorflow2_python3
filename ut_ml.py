#!/usr/bin/env python3
"""
TF2 docker unit test package for regression testing.

Make sure all the necessary tf python packages are bueno
"""

__author__ = 'Dave Turvene'
__email__  = 'dturvene at dahetral.com'
__copyright__ = 'Copyright (c) 2019 Dahetral Systems'
__version__ = '0.1'
__date__ = '20191105'

import sys
from os import path
import unittest
from pdb import set_trace as bp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf

def plt_setup():
    '''configure matplotlib'''
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}
    matplotlib.rc('font', **font)

    plt.style.use(['ggplot'])

def ut_plt():
    '''
    test plt display of images
    '''
    imgfiles=['test-image1.jpg', 'test-image2.png']

    fig = plt.figure(figsize=(8, 12))
    rows=2
    columns=1
    i=1
    for f in imgfiles:
        fpath=path.join('/data/TEST_IMAGES/', f)
        img = mpimg.imread(fpath)
        a=fig.add_subplot(rows,columns,i)
        a.set_title(f)
        i+=1
        plt.imshow(img)
    plt.tight_layout(h_pad=10.0, w_pad=10.0)
    plt.show()

class Ut(unittest.TestCase):
    def setUp(self):
        plt_setup()
    def tearDown(self):
        pass
    #@unittest.skip('good')
    def test1(self):
        '''basic sanity test'''
        print(tf.reduce_sum(tf.random.normal([1000,1000])))
    #@unittest.skip('good')        
    def test2(self):
        ut_plt()
    @unittest.skip('good')
    def test3(self):
        pass

if __name__ == '__main__':
    # exec(open('ut_ml.py').read())
    print(f'python ver={sys.version}')
    print(f'numpy ver={np.__version__}')
    print(f'seaborn ver={sns.__version__}')
    print(f'matplotlib ver={matplotlib.__version__}')
    print(f'tf ver={tf.__version__}')

    unittest.main(exit=False)
