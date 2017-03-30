import os
import sys
import fnmatch

sys.path.append('sptk')

import numpy as np

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0, ext_logf0
from converter import mcep2vec, mfcc2vec, pitch2vec

wavpath = 'target/'
datapath = 'target/'

EXP = 1e+6

if __name__ == '__main__':
    
    wavlist = []
    for file in os.listdir(wavpath):
        if fnmatch.fnmatch(file, '*.wav'):
            wavlist.append(file)
    
    #wavlist = ['test001.wav'] # debug
    
    for wname in wavlist:
        root, ext = os.path.splitext(wname)
        wname = wavpath + wname
        root = datapath + root
        
        rname = root + '.raw'
        mcname = root + '.mcep'
        lf0name = root + '.lfzero'
        mfname = root + '.mfc'
        pname = root + '.pitch'
        
        w2r(wname, rname)
        
        ext_mcep(rname, mcname)
        ext_logf0(rname, lf0name)
        ext_pitch(rname, pname)
        ext_mfcc(rname, mfname)
        
        mcep = mcep2vec(mcname)
        pitch = pitch2vec(pname)
        lf0 = pitch2vec(lf0name)
        mfcc = mfcc2vec(mfname)
        
        mcsave = root + 'mc.npy'
        mfsave = root + 'mf.npy'
        lf0save = root + 'lf0.npy'
        psave = root + 'p.npy'
        
        np.save(mcsave, mcep)
        np.save(mfsave, mfcc)
        np.save(lf0save, lf0)
        np.save(psave, pitch)
