import os
import sys
import fnmatch

import numpy as np

datapath = 'target/'
"""
mceplist = []
for file in os.listdir(datapath):
    if fnmatch.fnmatch(file, '*mf.npy'):
        mceplist.append(file)
"""
mceplist = ['target001mc.npy'] # debug

INF = 1e+6

FRAME_SIZE = 200
MCEP_SIZE = 40

trainmcep = np.ones((1, FRAME_SIZE, MCEP_SIZE)) * INF

if __name__ == '__main__':
    for mcepname in mceplist:
        root, ext = os.path.splitext(mcepname)
        root = root[:-2]
        
        tmpfrm = np.shape(trainmcep)[0] - 1
        inflen = int(np.shape(np.where(trainmcep[tmpfrm]==INF)[0])[0] / 40)
        
        mfcc = np.load(datapath + mcepname)
        if np.shape(mfcc)[0] < inflen:
            trainmcep[tmpfrm, -np.shape(mfcc)[0]:] = mfcc
            continue
        elif inflen > 0:
            trainmcep[tmpfrm, -inflen:] = mfcc[:inflen]
            mfcc = mfcc[inflen:]
        
        tmplen = np.shape(mfcc)[0] // FRAME_SIZE + 1
        tmpmfcc = np.ones((tmplen, FRAME_SIZE, MCEP_SIZE)) * INF
        for i in range(tmplen-1):
            tmpmfcc[i] = mfcc[:FRAME_SIZE]
            mfcc = mfcc[FRAME_SIZE:]
        tmpmfcc[-1, :np.shape(mfcc)[0]] = mfcc
        
        trainmcep = np.vstack((trainmcep, tmpmfcc))
    
    trainmcep[np.where(trainmcep==INF)] = 0
        
    np.save(datapath + 'trainmc.npy', trainmcep)
