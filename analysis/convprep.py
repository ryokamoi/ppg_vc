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
    """
    wavlist = []
    for file in os.listdir(wavpath):
        if fnmatch.fnmatch(file, '*.wav'):
            wavlist.append(file)
    """
    wavlist = ['target001.wav'] # debug
    
    for wname in wavlist:
        root, ext = os.path.splitext(wname)
        wname = wavpath + wname
        root = datapath + root
        
        rname = root + '.raw'
        mcname = root + '.mcep'
        f0name = root + '.fzero'
        
        w2r(wname, rname)
        
        ext_mcep(rname, mcname)
        ext_f0(rname, f0name)
        
        mcep = mcep2vec(mcname)
        f0 = pitch2vec(f0name)
        
        log = np.zeros(np.shape(f0))
        for i in range(np.shape(log)[0]):
            if f0[i] >= 1:
                log[i] = np.log(f0[i])
            else:
                log[i] = - EXP
        
        mcsave = root + 'mc' + '.npy'
        lf0save = root + 'lf0' + '.npy'
        
        np.save(mcsave, mcep)
        np.save(lf0save, log)
        