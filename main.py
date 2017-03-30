import os
import sys
import fnmatch

sys.path.append('sptk')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sptktools import w2r
from extract import ext_mcep, ext_mfcc, ext_pitch, ext_logf0
from converter import mcep2vec, mfcc2vec, pitch2vec, vec2mcep, vec2pitch, synthesize
from keras.models import load_model

wname = 'konino2.wav'
wavpath = 'main/'
datapath = 'main/'

if __name__ == '__main__':
    # extract features of the source file
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
    
    # SI-ASR
    FRAME_SIZE = 2000
    MFCC_SIZE = 40
    BATCH_SIZE = 32
    mfccname = root+'mf.npy'
    trainmfcc = np.load(mfccname)
    siasr_ss = joblib.load('standard.pkl') 
    trainmfcc = siasr_ss.transform(trainmfcc)
    trainmfcc = trainmfcc.reshape((1, -1, MFCC_SIZE))
    siasr_model = load_model('main/siasr_model.hdf5')
    probability = siasr_model.predict(trainmfcc, batch_size=BATCH_SIZE)
    np.save('main/resultppg.npy', probability)
    
    # PPGs+LogF0 to std MCEP
    PHONEME_SIZE = 36
    ppgname = 'main/resultppg.npy'
    lf0name = root+'lf0.npy'
    
    trainppg = np.load(ppgname)[0]
    trainlf0 = np.load(lf0name).reshape((-1, 1))
    nonzeroind = np.where(trainlf0>-1)
    notminus = trainlf0[nonzeroind].reshape(-1, 1)
    tmpss = StandardScaler().fit(notminus)
    standard = tmpss.transform(notminus)
    mean, std = np.load('lf0meanstd.npy')
    modified = standard*std+mean
    lf0_ss = joblib.load('standardlf0.pkl') 
    zero_one = lf0_ss.transform(modified)
    trainlf0[nonzeroind] = zero_one.reshape(-1)
    trainppglf0 = np.hstack((trainlf0, trainppg)).reshape(1, -1, PHONEME_SIZE+1)
    
    # save converted pitch
    SR = 16000
    fullpitch = np.zeros(np.shape(trainlf0))
    fullpitch[nonzeroind] = (SR/np.exp(modified)).reshape(-1)
    pitchname = 'main/resultpitch.npy'
    np.save(pitchname, fullpitch)
    
    ppg2mcep_model = load_model('main/ppgfl02stdmcep.hdf5')
    onezero_mcep = ppg2mcep_model.predict(trainppglf0, batch_size=BATCH_SIZE)
    mcep_ss = joblib.load('stdmcep.pkl') 
    adhock = StandardScaler().fit(onezero_mcep[0])
    trick = adhock.transform(onezero_mcep[0])
    trick = trick*0.9+adhock.mean_*0.9
    inversed_mcep = mcep_ss.inverse_transform(trick)
    np.save('main/resultmcep.npy', inversed_mcep)
    
    # synthesize_voice
    wavpath = 'main/'
    datapath = 'main/'
    savepath = 'main/'
    
    mcep = np.load('main/resultmcep.npy')
    pitch = np.load('main/resultpitch.npy')
    
    s_mname = 'main/result.mcep'
    s_pname = 'main/result.pitch'
    s_rname = savepath + 'result.raw'
    s_wname = savepath + 'result.wav'
    
    vec2mcep(mcep, s_mname)
    vec2pitch(pitch, s_pname)
    synthesize(s_pname, s_mname, s_rname, s_wname)
    