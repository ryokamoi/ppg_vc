import os
import sys

sys.path.append('sptk')

import numpy as np

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0
from converter import mcep2vec, mfcc2vec, pitch2vec

EXP = 1e+6

if __name__ == '__main__':
    wavlist = ['data/a01.wav']
    
    for wname in wavlist:
        root, ext = os.path.splitext(wname)
        
        rname = root + '.raw'
        mcname = root + '.mcep'
        f0name = root + '.fzero'
        mfname = root + '.mfc'
        pname = root + '.pitch'
        
        w2r(wname, rname)
        
        ext_mcep(rname, mcname)
        ext_f0(rname, f0name)
        ext_pitch(rname, pname)
        ext_mfcc(rname, mfname)
        
        mcep = mcep2vec(mcname)
        pitch = pitch2vec(pname)
        f0 = pitch2vec(pname)
        mfcc = mfcc2vec(mfname)
        
        log = np.zeros(np.shape(f0))
        for i in range(np.shape(log)[0]):
            if f0[i] >= 1:
                log[i] = np.log(f0[i])
            else:
                log[i] = - EXP
        
        mcsave = root + 'mc' + '.npy'
        mfsave = root + 'mf' + '.npy'
        lf0save = root + 'lf0' + '.npy'
        
        np.save(mcsave, mcep)
        np.save(mfsave, mfcc)
        np.save(lf0save, log)
