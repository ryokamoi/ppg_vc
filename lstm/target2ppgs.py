import os
import sys
import fnmatch

import numpy as np
import scipy.stats as sp
from sklearn.externals import joblib

datapath = 'target/'
mfcclist = []
for filename in os.listdir(datapath):
    if fnmatch.fnmatch(filename, '*mf.npy'):
        mfcclist.append(filename)
#mfcclist = ['test001mf.npy'] # debug
#mfcclist = mfcclist[:10] # debug

MFCC_SIZE = 40
BATCH_SIZE = 32

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import load_model

ss = joblib.load('standard.pkl') 
modelpath = 'lstm/newmodel0328_bi/weights21_del.hdf5'
model = load_model(modelpath)

for mfccname in mfcclist:
    print(mfccname)
    root, ext = os.path.splitext(mfccname)
    root = root[:-2]
    trainmfcc = np.load(datapath + mfccname)
    trainmfcc = ss.transform(trainmfcc)
    trainmfcc = trainmfcc.reshape((1, -1, MFCC_SIZE))
    ppgname = root + 'ppg.npy'
        
    probability = model.predict(trainmfcc, batch_size=BATCH_SIZE)
    np.save(datapath + ppgname, probability)
