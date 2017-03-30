import os
import sys
import fnmatch

sys.path.append('sptk')

import numpy as np

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0
from converter import mfcc2vec, pitch2vec

wavpath = 'segmentation-kit/wav/'
datapath = 'data/'

EXP = 1e+6

if __name__ == '__main__':
    wavlist = []
    for file in os.listdir(wavpath):
        if fnmatch.fnmatch(file, '*.wav'):
            wavlist.append(file)
    #wavlist = ['chi0001.wav'] # debug
    
    for wname in wavlist:
        root, ext = os.path.splitext(wname)
        wname = wavpath + wname
        root = datapath + root
        
        rname = root + '.raw'
        lf0name = root + '.lfzero'
        mfname = root + '.mfc'
        pname = root + '.pitch'
        
        w2r(wname, rname)
        
        ext_logf0(rname, lf0name)
        ext_pitch(rname, pname)
        ext_mfcc(rname, mfname)
        
        pitch = pitch2vec(pname)
        lf0 = pitch2vec(lf0name)
        mfcc = mfcc2vec(mfname)
                
        mfsave = root + 'mf' + '.npy'
        lf0save = root + 'lf0' + '.npy'
        
        np.save(mfsave, mfcc)
        np.save(lf0save, lf0)
        