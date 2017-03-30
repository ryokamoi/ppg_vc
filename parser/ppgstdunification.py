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

#mceplist = ['target001mc.npy'] # debug

INF = 1e+6

FRAME_SIZE = 2000
PHONEME_SIZE = 36
MCEP_SIZE = 40

trainmcep = np.zeros((1, MCEP_SIZE))
trainppg = np.zeros((1, PHONEME_SIZE))

for mcepname in mceplist:
    root, ext = os.path.splitext(mcepname)
    root = root[:-2]
    ppgname = root + 'ppg.npy'
    tmpmcep = np.load(datapath + mcepname)
    tmpppg = np.load(datapath + ppgname)[0]
    trainmcep = np.vstack((trainmcep, tmpmcep))
    trainppg = np.vstack((trainppg, tmpppg))

trainmcep = trainmcep[1:]
ss = StandardScaler().fit(trainmcep)
stdtrainmcep = ss.transform(trainmcep)
joblib.dump(ss, 'stdmcep.pkl') 
trainppg = trainppg[1:]

#np.save(datapath + 'flattenmfc.npy', trainmcep)
#np.save(datapath + 'flattenppg.npy', trainppg)

mceppad = np.zeros((FRAME_SIZE-np.shape(trainmcep)[0]%FRAME_SIZE, MCEP_SIZE))
ppgpad = np.zeros((FRAME_SIZE-np.shape(trainppg)[0]%FRAME_SIZE, PHONEME_SIZE))
stdtrainmcep = np.vstack((stdtrainmcep, mceppad))
trainppg = np.vstack((trainppg, ppgpad))

stdtrainmcep = stdtrainmcep.reshape((-1, FRAME_SIZE, MCEP_SIZE))
trainppg = trainppg.reshape((-1, FRAME_SIZE, PHONEME_SIZE))

np.save(savepath + 'targetstdmc.npy', stdtrainmcep)
np.save(savepath + 'targetppg.npy', trainppg)


"""
trainmcep = np.ones((1, FRAME_SIZE, MCEP_SIZE)) * INF

if __name__ == '__main__':
    for mcepname in mceplist:
        root, ext = os.path.splitext(mcepname)
        root = root[:-2]
        
        tmpfrm = np.shape(trainmcep)[0] - 1
        inflen = int(np.shape(np.where(trainmcep[tmpfrm]==INF)[0])[0] / 40)
        
        mcep = np.load(datapath + mcepname)
        if np.shape(mcep)[0] < inflen:
            trainmcep[tmpfrm, -np.shape(mcep)[0]:] = mcep
            continue
        elif inflen > 0:
            trainmcep[tmpfrm, -inflen:] = mcep[:inflen]
            mcep = mcep[inflen:]
        
        tmplen = np.shape(mcep)[0] // FRAME_SIZE + 1
        tmpmcep = np.ones((tmplen, FRAME_SIZE, MCEP_SIZE)) * INF
        for i in range(tmplen-1):
            tmpmcep[i] = mcep[:FRAME_SIZE]
            mcep = mcep[FRAME_SIZE:]
        tmpmcep[-1, :np.shape(mcep)[0]] = mcep
        
        trainmcep = np.vstack((trainmcep, tmpmcep))
    
    trainmcep[np.where(trainmcep==INF)] = 0
        
    np.save(datapath + 'trainmc.npy', trainmcep)
"""
