import os
import sys
import fnmatch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

datapath = 'target/'
savepath = 'targetdata/'

mceplist = []
for file in os.listdir(datapath):
    if fnmatch.fnmatch(file, '*mc.npy'):
        mceplist.append(file)

INF = 1e+6

FRAME_SIZE = 2000
PHONEME_SIZE = 36
MCEP_SIZE = 40

trainmcep = np.zeros((1, MCEP_SIZE))
trainppg = np.zeros((1, PHONEME_SIZE))
trainlf0 = np.zeros((1, 1))

for mcepname in mceplist:
    root, ext = os.path.splitext(mcepname)
    root = root[:-2]
    ppgname = root + 'ppg.npy'
    lf0name = root + 'lf0.npy'
    tmpmcep = np.load(datapath + mcepname)
    tmpppg = np.load(datapath + ppgname)[0]
    trainmcep = np.vstack((trainmcep, tmpmcep))
    trainppg = np.vstack((trainppg, tmpppg))
    tmplf0 = np.load(datapath + lf0name).reshape(-1, 1)
    trainlf0 = np.vstack((trainlf0, tmplf0))
    assert np.shape(tmpppg)[0] == np.shape(tmplf0)[0]

trainmcep = trainmcep[1:]
trainppg = trainppg[1:]
trainlf0 = trainlf0[1:]
notminus = trainlf0[np.where(trainlf0>-1)]
ss = StandardScaler().fit(notminus.reshape(-1, 1))
zero_one = ss.transform(notminus.reshape(-1, 1))
trainlf0[np.where(trainlf0>-1)] = zero_one.reshape(-1)
joblib.dump(ss, 'standardlf0.pkl') 
trainppglf0 = np.hstack((trainlf0, trainppg))

mceppad = np.zeros((FRAME_SIZE-np.shape(trainmcep)[0]%FRAME_SIZE, MCEP_SIZE))
ppglf0pad = np.zeros((FRAME_SIZE-np.shape(trainppg)[0]%\
                                    FRAME_SIZE, PHONEME_SIZE+1))
trainmcep = np.vstack((trainmcep, mceppad))
trainppglf0 = np.vstack((trainppglf0, ppglf0pad))

trainmcep = trainmcep.reshape((-1, FRAME_SIZE, MCEP_SIZE))
trainppglf0 = trainppglf0.reshape((-1, FRAME_SIZE, PHONEME_SIZE+1))

np.save(savepath + 'targetmc.npy', trainmcep)
np.save(savepath + 'targetppglf0.npy', trainppglf0)
