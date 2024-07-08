from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')
import os
import keras

print("Keras = {}".format(keras.__version__))
import tensorflow as tf
import tensorlayer as tl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import pylab
import sys
import math
import keras.backend.tensorflow_backend as KTF
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import shutil
from sklearn import metrics
import random
from random import shuffle
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from model_logic2 import *
from generator0 import *
import pickle
from keras.utils import plot_model
from HRNet_1D import ResNet50

os.environ["CUDA_VISIBLE_DEVICES"]="0"     #GPU

from tensorflow import ConfigProto, GPUOptions, Session
from keras.backend.tensorflow_backend import set_session

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    set_session(Session(config=ConfigProto(gpu_options=GPUOptions(allow_growth=True))))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
sys.setrecursionlimit(2000)

training_data_path = "/data/TrainingData"          # The path training set, if use this network, the training set should be generated first
training_gooddata_path = "/data/TrainingGoodData"
validation_data_path = "/data/ValidationData"
validation_gooddata_path = "/data/ValidationGoodData"

x_train = tl.files.load_file_list(path=training_data_path,   #The format of the data is .txt, which can be changed according to your needs
                                  regx='.*.txt',
                                  printable=False)
x_train.sort(key=tl.files.natural_keys)

y_train = tl.files.load_file_list(path=training_gooddata_path,
                                  regx='.*.txt',
                                  printable=False)
y_train.sort(key=tl.files.natural_keys)

x_val = tl.files.load_file_list(path=validation_data_path,
                                regx='.*.txt',
                                printable=False)
x_val.sort(key=tl.files.natural_keys)

y_val = tl.files.load_file_list(path=validation_gooddata_path,
                                regx='.*.txt',
                                printable=False)
y_val.sort(key=tl.files.natural_keys)  ##

train_all_num = len(x_train)
val_all_num = len(x_val)
f_xtrain = []
f_ytrain = []
f_xval = []
f_yval = []

for i in range(train_all_num):
    f_xtrain.append(os.path.join(training_data_path, x_train[i]))
    f_ytrain.append(os.path.join(training_gooddata_path, y_train[i]))

for i in range(val_all_num):
    f_xval.append(os.path.join(validation_data_path, x_val[i]))
    f_yval.append(os.path.join(validation_gooddata_path, y_val[i]))

parser = OptionParser()  ##parameters of training
parser.add_option("--run", dest="run", help="the index of gpu are used", default=1, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=4096, type="int")   #point of data
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=1, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=1, type="int")
parser.add_option("--nb_class", dest="nb_class", help="number of class", default=1, type="int")
parser.add_option("--verbose", dest="verbose", help="verbose", default=2, type="int")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")

(options, args) = parser.parse_args()

model_path_idx = options.run
model_path = "./model/EDHR_Net_" + str(model_path_idx) + "/"  ##保存模型

if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

class setup_config():
    ##parameters of training
    optimizer = "Adam"
    lr = 1e-3
    GPU_COUNT = 1
    nb_epoch = 15
    patience = 15

    def __init__(self,
                 input_rows=4096,
                 input_cols=1,
                 input_deps=1,
                 batch_size=32,
                 verbose=2,
                 nb_class=1,
                 ):
        self.exp_name = "EDHR_Net"
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.nb_class = nb_class
        if nb_class > 1:
            self.activation = "LeakyReLU"
        else:
            self.activation = "LeakyReLU"

    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and "ids" not in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

config = setup_config(
    input_rows=options.input_rows,
    input_cols=options.input_cols,
    input_deps=options.input_deps,
    batch_size=options.batch_size,
    verbose=options.verbose,
    nb_class=options.nb_class,
)
config.display()

model = ResNet50(config.input_rows, config.input_cols, config.input_deps, config.nb_class)

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)

model.compile(optimizer=opt,
              loss=MAE,
              metrics=['mse']
              )

model.summary()

if os.path.exists(os.path.join(model_path, config.exp_name + ".txt")):
    os.remove(os.path.join(model_path, config.exp_name + ".txt"))
with open(os.path.join(model_path, config.exp_name + ".txt"), 'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(logs_path, config.exp_name)):
    os.makedirs(os.path.join(logs_path, config.exp_name))

tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                         )
tbCallBack.set_model(model)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=config.patience,
                                               verbose=0,
                                               mode='min',
                                               )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name + "{epoch:03d}.h5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              mode='min',
                                              )
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto')

callbacks = [check_point, early_stopping, tbCallBack, reduce_lr]

while config.batch_size > 0:
    try:
        history = model.fit_generator(data_generator(f_xtrain, f_ytrain, config.batch_size),
                                      steps_per_epoch=len(f_xtrain) // config.batch_size,
                                      epochs=config.nb_epoch,
                                      verbose=config.verbose,
                                      validation_data=data_generator(f_xval, f_yval, 2),
                                      validation_steps=len(f_xval) // 2,
                                      callbacks=callbacks)
        break
    except tf.errors.ResourceExhaustedError as e:
        config.batch_size = int(config.batch_size / 2.0)
        print("\n> Batch size = {}".format(config.batch_size))
