import os
import sys
import fnmatch

sys.path.append('sptk')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0, ext_logf0
from converter import mcep2vec, mfcc2vec, pitch2vec, vec2pitch

lf0path = 'target/'
datapath = 'target/'
sourcefile = 'data/a01lf0.npy'

SR = 16000
EXP = 1e+6

if __name__ == '__main__':
    """
    lf0list = []
    for file in os.listdir(lf0path):
        if fnmatch.fnmatch(file, '*lf0.npy'):
            lf0list.append(file)
    
    targetlf0 = np.array([0])
    for fl0name in lf0list:
        tmpfl0 = np.load(lf0path+fl0name)
        targetlf0 = np.hstack((targetlf0, tmpfl0[np.where(tmpfl0>-1)]))
    
    meanstd = np.array([targetlf0.mean(), targetlf0.std()])
    np.save('meanstd.npy', meanstd)
    """
    meanstd = np.load('meanstd.npy')
    mean = meanstd[0]
    std = meanstd[1]
    
    source = np.load(sourcefile)
    notminus = source[np.where(source>-1)]
    ss = StandardScaler().fit(notminus.reshape(-1, 1))
    zero_one = ss.transform(notminus.reshape(-1, 1))
    same_ms = zero_one.reshape(-1) * std + mean
    
    non0f0 = np.exp(same_ms)
    non0pitch = SR / non0f0
    fullpitch = np.zeros(np.shape(source))
    fullpitch[np.where(source>-1)] = non0pitch.reshape(-1)
    
    pitchname = 'result/result.pitch'
    vec2pitch(fullpitch, pitchname)
        