import warnings
warnings.filterwarnings('ignore')
import os
import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import math
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
import scipy.io as sio

def MAE(y_true, y_pred):
  a=K.mean(K.abs(y_true- y_pred))
  return a

indirect1=4096    # the point of data
batch_size=1

model_path = "./model/EDHR_Net_1/"   # the path of model

print('[*] load data ... ')
# under=sio.loadmat('exp_ibuprofen.mat')        # the data to be processed
under=sio.loadmat('exp_kanamycin.mat')
# under=sio.loadmat('exp_estradiol.mat')
data=under['data']
X_test=data['input_x']
X_test=np.array(X_test[0,0],dtype=np.float32)
X_test=np.reshape(X_test, (batch_size,indirect1, 1, 1))

################################################

print('[*] define model ...')

model = load_model(os.path.join("EDHR_Net.h5"), custom_objects={'MAE':MAE})

print('[*] start testing ...')

x_gen = model.predict([X_test], batch_size=batch_size, verbose=1)
x_gen=np.array(x_gen,dtype=np.float32)

sio.savemat('rec_spec_weak.mat',{'rec_spec':x_gen,'X_test':X_test})   # the reconstructed data

print("[*] Job finished!")





