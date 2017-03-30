import os
import sys
import fnmatch

import numpy as np
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

datapath = 'data/'
savepath = 'traindata/'

mfcclist = []
for filename in os.listdir(datapath):
    if fnmatch.fnmatch(filename, '*mf.npy'):
        mfcclist.append(filename)

# mfcclist = ['a01mf.npy'] # debug
# mfcclist = mfcclist[:3] # debug

INF = 1e+6

FRAME_SIZE = 2000
MFCC_SIZE = 40
PHONEME_SIZE = 36

trainmfcc = np.zeros((1, MFCC_SIZE))
trainppg = np.zeros((1, PHONEME_SIZE))

for mfccname in mfcclist:
    root, ext = os.path.splitext(mfccname)
    root = root[:-2]
    ppgname = root + 'ppg.npy'
    tmpmfcc = np.load(datapath + mfccname)
    tmpppg = np.load(datapath + ppgname)
    trainmfcc = np.vstack((trainmfcc, tmpmfcc))
    trainppg = np.vstack((trainppg, tmpppg))

trainmfcc = trainmfcc[1:]
ss = StandardScaler()
ss.fit(trainmfcc)
trainmfcc = ss.transform(trainmfcc)
joblib.dump(ss, 'standard.pkl') 
#trainmfcc = sp.stats.zscore(trainmfcc, axis=0, ddof=1)
#trainmfcc = sp.stats.zscore(trainmfcc, axis=0, ddof=1)+0.5
trainppg = trainppg[1:]

#np.save(datapath + 'flattenmfc.npy', trainmfcc)
#np.save(datapath + 'flattenppg.npy', trainppg)

mfccpad = np.zeros((FRAME_SIZE-np.shape(trainmfcc)[0]%FRAME_SIZE, MFCC_SIZE))
ppgpad = np.zeros((FRAME_SIZE-np.shape(trainppg)[0]%FRAME_SIZE, PHONEME_SIZE))
trainmfcc = np.vstack((trainmfcc, mfccpad))
trainppg = np.vstack((trainppg, ppgpad))

trainmfcc = trainmfcc.reshape((-1, FRAME_SIZE, MFCC_SIZE))
trainppg = trainppg.reshape((-1, FRAME_SIZE, PHONEME_SIZE))

np.save(savepath + 'trainmfc.npy', trainmfcc)
np.save(savepath + 'trainppg.npy', trainppg)

"""
trainmfcc = np.ones((1, FRAME_SIZE, MFCC_SIZE)) * INF
trainppg = np.ones((1, FRAME_SIZE, PHONEME_SIZE)) * INF

if __name__ == '__main__':
    for mfccname in mfcclist:
        root, ext = os.path.splitext(mfccname)
        root = root[:-2]
        ppgname = root + 'ppg.npy'
        
        tmpfrm = np.shape(trainmfcc)[0] - 1
        inflen = int(np.shape(np.where(trainmfcc[tmpfrm]==INF)[0])[0] / 40)
        
        mfcc = np.load(datapath + mfccname)
        ppg = np.load(datapath + ppgname)
        if np.shape(mfcc)[0] < inflen:
            trainmfcc[tmpfrm, -np.shape(mfcc)[0]:] = mfcc
            trainppg[tmpfrm, -np.shape(mfcc)[0]:] = ppg
            continue
        elif inflen > 0:
            trainmfcc[tmpfrm, -inflen:] = mfcc[:inflen]
            trainppg[tmpfrm, -inflen:] = ppg[:inflen]
            mfcc = mfcc[inflen:]
            ppg = ppg[inflen:]
        
        tmplen = np.shape(mfcc)[0] // FRAME_SIZE + 1
        tmpmfcc = np.ones((tmplen, FRAME_SIZE, MFCC_SIZE)) * INF
        tmpppg = np.ones((tmplen, FRAME_SIZE, PHONEME_SIZE)) * INF
        for i in range(tmplen-1):
            tmpmfcc[i] = mfcc[:FRAME_SIZE]
            tmpppg[i] = ppg[:FRAME_SIZE]
            mfcc = mfcc[FRAME_SIZE:]
            ppg = ppg[FRAME_SIZE:]
        tmpmfcc[-1, :np.shape(mfcc)[0]] = mfcc
        tmpppg[-1, :np.shape(ppg)[0]] = ppg
        
        trainmfcc = np.vstack((trainmfcc, tmpmfcc))
        trainppg = np.vstack((trainppg, tmpppg))
    
    trainmfcc[np.where(trainmfcc==INF)] = 0
    trainppg[np.where(trainppg==INF)] = 0
        
    #np.save(datapath + 'trainmfc.npy', trainmfcc)
    #np.save(datapath + 'trainppg.npy', trainppg)
    np.save('trainmfc.npy', trainmfcc) #debug
    np.save('trainppg.npy', trainppg) #debug
"""
