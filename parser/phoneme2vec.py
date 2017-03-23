import os
import sys
import fnmatch

import numpy as np

wavpath = 'segmentation-kit/wav/'
labpath = 'segmentation-kit/wav/'
datapath = 'data/'
"""
lablist = []
for file in os.listdir(wavpath):
    if fnmatch.fnmatch(file, '*.lab'):
        lablist.append(file)
"""
lablist = ['a01.lab'] # debug

PHONEME = 36

f = open('phoneme.txt')
phonemelist = []
for phoneme in f.readlines():
    phonemelist.append(phoneme.replace('\n', ''))
f.close()
phonemedic = {p: phonemelist.index(p) for p in phonemelist}
phonemedic['a:'] = phonemedic['a'] # adhoc
phonemedic['i:'] = phonemedic['i'] # adhoc
phonemedic['u:'] = phonemedic['u'] # adhoc
phonemedic['e:'] = phonemedic['e'] # adhoc
phonemedic['o:'] = phonemedic['o'] # adhoc
phonemedic['sp'] = phonemedic['sil'] # adhoc
phonemedic['silB'] = phonemedic['sil'] # adhoc
phonemedic['silE'] = phonemedic['sil'] # adhoc

for labname in lablist:
    root, ext = os.path.splitext(labname)
    
    mfcc = np.load(datapath + root + 'mf.npy')
    timelen = np.shape(mfcc)[0]
    ppg = np.zeros(timelen)
    
    f = open(labpath + labname)
    for line in f.readlines():
        lablist = line.split(' ')
        start = int(float(lablist[0]) // 0.005)
        ppg[start:] = phonemedic[lablist[2].replace('\n', '')]
    f.close()
    
    onehot = np.zeros((timelen, PHONEME))
    onehot[range(timelen), ppg.astype(int)] = 1
    
    np.save(datapath + root + 'ppg.npy', onehot)