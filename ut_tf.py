#!/usr/bin/env python3
"""
unit test for packages and keras support

* https://www.tensorflow.org/tutorials/keras/regression

220401: a lot of code has been updated and streamlined

"""

import sys
from pdb import set_trace as bp
import unittest

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import timeit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def get_rawdata():
    '''get raw dataset
       cached in ~/.keras/datasets/auto-mpg.data
    398 records
MPG:
Cylinders: 8, 6, 4
Displacement: engine in CI
HP:
Weight:
Accel:
Model Year:
Origin: 1=USA, 2=Europe, 3=Japan
    '''
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values = "?", comment='\t',
                              sep=" ", skipinitialspace=True)

    return(raw_dataset)

def getdata():
    '''
    retrieve the raw dataframe
    clean
    convert origin strings to one-hot fields
    split df into train and test
    get training stats
    '''

    # Pandas DataFrame
    rd = get_rawdata()
    
    # why do this?
    dataset = rd.copy()
    # print('dataset tail\n', dataset.tail())

    # bad data fields, remove row (398 to 392)
    print('isna:\n{}'.format(dataset.isna().sum()))
    dataset = dataset.dropna()

    # create USA/Europe/Japan one-hot columns
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    # split data into train and test datasets
    # return a random sample from axis of object, default is stat axis
    # 314
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    # 78
    test_dataset = dataset.drop(train_dataset.index)

    # get useful statistics about training set
    # removing the label (MPG) because it is the target
    # transpose: flip rows/columns so each feature is on a row
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    # print('train_stats:\n', train_stats)

    return (train_dataset, test_dataset, train_stats)

def view_sns(tr_d, figfile='/data/TEST_IMAGES/reg1.png'):
    '''tr_d is a pd.DataFrame'''

    print('create pairplot')
    diag=sns.pairplot(
        tr_d[["MPG", "Cylinders", "Displacement", "Weight", "Acceleration"]],
        diag_kind="kde")

    print(f'save image to {figfile}...')
    diag.savefig(figfile)
    print('done save')

def view_pd(tr_d):
    '''tr_d is a pd.DataFrame'''

    print(tr_d[:5])
    
    tr_np=tr_d.to_numpy()

def build_model_relu_3(feature_num=9):
    '''
    feature_num=9
    MSE/MAE: 6.34/1.97
    MSE/MAE: 6.02/1.93
    '''
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[feature_num]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # RMSprop algorithm - moving average of the square of gradients
    # divide gradient by root of the average
    # 0.001 is the learning rate
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # configure the model for training
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def plot_history(history):
    '''
    show MPG and MPG^2 training for all epochs
    '''
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # show last 5 entries of training history, max is 1000
    # early_stop usually 20-70 epochs for Mean Abs Error <2.0 MPG
    # print('hist.tail:\n', hist.tail())
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    #plt.show()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()

    # show both MPG and MPG^2 graphs
    plt.show()

def ut_keras_br():
    '''
    220401: bad link
    * https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_regression.ipynb
    moved to 
    * https://www.tensorflow.org/tutorials/keras/regression
    * https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb
    '''
    # training, test, training stats PD DataFrames
    (tr_data, ts_data, tr_stats) = getdata()

    # save sns png file
    # view_sns(ts_data)

    # bp()

    # remove MPG column and create in separate label objects
    print('remove MPG')
    tr_labels = tr_data.pop('MPG')
    ts_labels = ts_data.pop('MPG')

    # normalize by diff of value and mean divided by standard deviation
    tr_norm = (tr_data - tr_stats['mean'])/ tr_stats['std']
    ts_norm = (ts_data - tr_stats['mean'])/ tr_stats['std']

    # build the model using the total number of features
    model = build_model_relu_3( len(tr_norm.keys()) )

    # model.summary(print_fn=print)
  
    # stop when loss not improving
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # fit to normalized training data and labels
    history = model.fit(tr_norm,
                        tr_labels,
                        epochs=1000,
                        validation_split = 0.2,
                        verbose=0,
                        callbacks=[early_stop, PrintDot()])
    # new line after PrintDot calls...
    print('\n')

    # show plot of epochs to MAE and MSE
    # plot_history(history)

    # now run model on normalized test dataset and see loss
    loss, mae, mse = model.evaluate(ts_norm, ts_labels, verbose=0)
    print("Testset MPG LOSS:{:5.2f} MAE:{:5.2f} MSE:{:5.2f}".format(loss, mae, mse))

    # now use model to predict MPG for test set and compare with real labels
    ts_preds = model.predict(ts_norm).flatten()
    # bp()
    plt.scatter(ts_labels, ts_preds)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = ts_preds - ts_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

def ut_tfbeg():
    '''
    Demo TF2 beginner test using mnist
    See https://www.tensorflow.org/tutorials/quickstart/beginner
    '''
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('fit')
    model.fit(x_train, y_train, epochs=5)
    print('evaluate')
    scores=model.evaluate(x_test, y_test, verbose=0)
    print('loss={} acc={}'.format(scores[0], scores[1]))

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
    #@unittest.skip('pass')
    def test1(self):
        pass
    #@unittest.skip('good')
    def test2(self):
        ut_keras_br()
    @unittest.skip('good')        
    def test3(self):
        ut_tfbeg()

if __name__ == '__main__':
    # exec(open('./ut_tf.py').read())
    print('tf={}'.format(tf.__version__))
    unittest.main(exit=False)
