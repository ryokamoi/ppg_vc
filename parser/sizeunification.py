import os
import sys
import fnmatch

import numpy as np

datapath = 'data/'
"""
mfcclist = []
for file in os.listdir(datapath):
    if fnmatch.fnmatch(file, '*mf.npy'):
        mfcclist.append(file)
"""
mfcclist = ['a01mf.npy'] # debug

INF = 1e+6

FRAME_SIZE = 200
MFCC_SIZE = 40
PHONEME_SIZE = 36

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
        if inflen > 0:
            trainmfcc[tmpfrm, -inflen:] = mfcc[:inflen]
            trainppg[tmpfrm, -inflen:] = ppg[:inflen]
            mfcc = mfcc[inflen:]
            ppg = ppg[inflen:]
        
        tmplen = np.shape(mfcc)[0] // 200 + 1
        tmpmfcc = np.ones((tmplen, FRAME_SIZE, MFCC_SIZE)) * INF
        tmpppg = np.ones((tmplen, FRAME_SIZE, PHONEME_SIZE)) * INF
        for i in range(tmplen-1):
            tmpmfcc[i] = mfcc[:200]
            tmpppg[i] = ppg[:200]
            mfcc = mfcc[200:]
            ppg = ppg[200:]
        tmpmfcc[-1, :np.shape(mfcc)[0]] = mfcc
        tmpppg[-1, :np.shape(ppg)[0]] = ppg
        
        trainmfcc = np.vstack((trainmfcc, tmpmfcc))
        trainppg = np.vstack((trainppg, tmpppg))
    
    trainmfcc[np.where(trainmfcc==INF)] = 0
    trainppg[np.where(trainppg==INF)] = 0
        
    np.save(datapath + 'trainmfc.npy', trainmfcc)
    np.save(datapath + 'trainppg.npy', trainppg)
