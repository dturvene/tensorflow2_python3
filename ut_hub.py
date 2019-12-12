#!/usr/bin/env python
"""
Model flowers using a pre-trained mobilenetv2 model from tensorflow_hub

* old but working
https://medium.com/towards-artificial-intelligence/testing-tensorflow-lite-image-classification-model-e9c0100d8de3
"""

import os
import unittest
import pathlib
from pdb import set_trace as bp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow_hub as hub

########################## globals #############################
# -1
AUTOTUNE = tf.data.experimental.AUTOTUNE

# match names against predict cross-entropy values
# must match class order from stats
# flower_names = {0:'daisy', 1:'dandelion', 2:'rose', 3:'sunflower', 4:'tulip'}
flower_names = {0:'dandelion', 1:'daisy', 2:'sunflowers', 3:'roses', 4:'tulips'}

class Dflg1():
    '''
    https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb
    '''
    
    def __init__(self, epochs=10):

        # image size for decode_img and model input shape
        self.img_height = 224
        self.img_width = 224
        # batchsize for dataset
        self.batch_size = 32

        # train datasets
        self.epochs=epochs
        
        # name for save/restore model
        # https://www.tensorflow.org/guide/saved_model
        # self.save_dir = '/home/dturvene/ML_DATA/saved_models'
        # change for docker volume
        self.save_dir = '/data/saved_models'
        # need to explicitly create directories before calling model.save
        self.model_name = 'dflg1/3/dflg1.h5'

    def load_url(self):
        '''
        get flowers to ~/.keras/datasets/flower_photos
        '''
        self.data_dir = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            fname='flower_photos',
            untar=True)

        # convert to PosixPath instance for searches/globbing
        posixpath = pathlib.Path(self.data_dir)
        self.image_count = len(list(posixpath.glob('*/*.jpg')))
        self.class_names = np.array([item.name for item in posixpath.glob('*') if item.name != 'LICENSE.txt'])

        # used for fit on generated data
        self.steps_per_epoch=np.ceil(self.image_count/self.batch_size)

    def idg_load(self):
        datagen_kwargs = dict(rescale=1./255, validation_split=0.20)
        image_generator = ImageDataGenerator(**datagen_kwargs)

        dataflow_kwargs = dict(target_size=(self.img_width, self.img_height),
                               batch_size=self.batch_size,
                               interpolation='bilinear',
                               classes = list(self.class_names))

        # Found 731 images belonging to 5 classes.
        self.val_gen = image_generator.flow_from_directory(str(self.data_dir),
                                                           subset='validation',
                                                           shuffle=False,
                                                           **dataflow_kwargs)

        # Found 2939 images belonging to 5 classes.
        if False:
            train_generator = ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagen_kwargs)
        else:
            train_generator = image_generator
            
        self.train_gen = train_generator.flow_from_directory(str(self.data_dir),
                                                             subset='training',
                                                             shuffle=True,
                                                             **dataflow_kwargs)
    def m1_bld(self):
        _URL='https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'

        self.model = models.Sequential()
        self.model.add( hub.KerasLayer(_URL, trainable=False) )
        self.model.add(layers.Dropout(0.2))
        self.model.add( layers.Dense(self.train_gen.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001)) )
        self.model.build((None,)+(self.img_height, self.img_width)+(3,))
        #self.model.summary()
        
        if True:
            self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                               optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
                               metrics=['accuracy'])
        else:
            self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                               optimizer=tf.keras.optimizers.Adam(lr=0.005, epsilon=0.1),
                               metrics=['accuracy'])

    def m1_train(self):
        steps_per_epoch = self.train_gen.samples//self.train_gen.batch_size
        val_steps_per_epoch = self.val_gen.samples//self.val_gen.batch_size

        self.hist = self.model.fit_generator(
            self.train_gen,
            epochs=self.epochs,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_gen,
            validation_steps=val_steps_per_epoch).history

        # Measure accuracy and loss after training
        final_loss, final_accuracy = self.model.evaluate(self.val_gen, steps = val_steps_per_epoch)
        print("Final loss: {:.2f}".format(final_loss))
        print("Final accuracy: {:.2f}%".format(final_accuracy * 100))
        
    def save_model(self):
        '''
        save model persistently in HDF5 format
        https://www.tensorflow.org/guide/saved_model
        export to SavedModel
        https://www.tensorflow.org/guide/keras/save_and_serialize
        '''
        model_path = os.path.join(self.save_dir, self.model_name)
        #tf.saved_model.save(self.model, model_path)
        #self.model.save(model_path, save_format='tf')
        # simplest, recommended way
        self.model.save(model_path)
        print('Saved model at {}'.format(model_path))
        
    def restore_model(self):
        '''
        restore model persistently
        https://www.tensorflow.org/guide/saved_model
        recreate the same model
        https://www.tensorflow.org/guide/keras/save_and_serialize
        '''
        model_path = os.path.join(self.save_dir, self.model_name)
        # self.model=tf.keras.models.load_model('/tmp/k1.2.h5', custom_objects={'KerasLayer':hub.KerasLayer})
        self.model=tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
        print('Restored trained model from {}'.format(model_path))

    def check_model(self):
        self.model.summary()
        print(list(self.model.signatures.keys()))
        infer=self.model.signatures['server_default']
        print(infer.structured_outputs)

    def stats(self):
        '''show stats'''
        print(f'total images={self.image_count}, flower classes={self.class_names}')
        for class_name in self.class_names:
            path=class_name+'/*.jpg'
            posixpath = pathlib.Path(self.data_dir)
            class_images = len(list(posixpath.glob(path)))
            print(f'images for {class_name}: {class_images}')

    def predict(self):
        '''
        predict new flowers against model
        images must be transformed the same those for training the model
        '''

        # dir_predict = self.save_dir + '/flowers.predict'
        self.dir_predict = '/data/flowers.predict'
        for fname in os.listdir(self.dir_predict):
            img = load_img( os.path.join(self.dir_predict, fname), target_size=(self.img_width, self.img_height) )
            img = img_to_array(img)
            img = img.reshape(1, self.img_width, self.img_height, 3)
            img = img.astype('float32')
            img = img/255.0
            rp = self.model.predict(img, verbose=0)
            print('fname={}: pred={} weights={}'.format(fname, flower_names[rp.argmax()], [i for i in rp[0]]))

    def plot_hist(self):
        '''
        '''
        plt.figure()
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(self.hist["loss"])
        plt.plot(self.hist["val_loss"])

        plt.figure()
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(self.hist["accuracy"])
        plt.plot(self.hist["val_accuracy"])
        
        plt.show()

class Ut(unittest.TestCase):
    
    def setUp(self):
        '''
        x = Ut()
        x.setUp()
        '''
        self.ut = Dflg1(epochs=5)
        self.ut.load_url()
        self.ut.stats()
        
    def tearDown(self):
        pass

    #@unittest.skip('good')
    def test_b1(self):
        self.ut.idg_load()
        self.ut.m1_bld()
        self.ut.m1_train()
        self.ut.predict()
        self.ut.save_model()

    #@unittest.skip('good')
    def test_p1(self):
        '''
        x.test_p1()
        '''
        self.ut.restore_model()
        self.ut.predict()

    def helper_show(self):
        '''helper to show first three batches of data'''
        for it in range(3):
            image_batch, label_batch=next(self.ut.train_gen)
            self.ut.show_batch(image_batch, label_batch)

if __name__ == '__main__':
    # exec(open('./ut_hub.py').read())
    print('tf={}, keras={}'.format(tf.__version__, tf.keras.__version__))
    print('hub={}'.format(hub.__version__))
    unittest.main(exit=False)
