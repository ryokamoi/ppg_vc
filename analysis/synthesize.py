import os
import sys

sys.path.append('sptk')

import numpy as np

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0
from converter import mcep2vec, pitch2vec, vec2mcep, vec2pitch, synthesize

wavpath = 'result/'
datapath = 'data/'
savepath = 'result/'

if __name__ == '__main__':
    wavlist = ['a01.wav']
    
    for wname in wavlist:
        root, ext = os.path.splitext(wname)
        save = savepath + root
        root = datapath + root
        
        rname = root + '.raw'
        mcname = root + '.mcep'
        f0name = root + '.fzero'
        mfname = root + '.mfc'
        pname = root + '.pitch'
        
        mcsave = root + 'mc' + '.npy'
        mfsave = root + 'mf' + '.npy'
        lf0save = root + 'lf0' + '.npy'
        
        mcep = mcep2vec(mcname)
        pitch = pitch2vec(pname)
        
        s_mname = root + '.mcep'
        s_pname = root + '.pitch'
        s_rname = savepath + '.raw'
        s_wname = savepath + '.wav'
        
        vec2mcep(mcep, s_mname)
        vec2pitch(pitch, s_pname)
        synthesize(s_pname, s_mname, s_rname, s_wname)
