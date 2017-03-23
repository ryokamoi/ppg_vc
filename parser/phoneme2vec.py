import os
import sys

import numpy as np

labpath = 'segmentation-kit/wav/'
datapath = 'data/'

"""
lablist = []
for file in os.listdir(wavpath):
    if fnmatch.fnmatch(file, '*.lab'):
        wavlist.append(file)
"""
lablist = ['a01.lab'] # debug

f = open('phoneme.txt')
phonemelist = []
for phoneme in f.readlines():
    phonemelist.append(phoneme.replace('\n', ''))
f.close()
phonemedic = {p: phonemelist.index(p) for p in phonemelist}
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
    
    np.save(datapath + root + 'ppg.npy', ppg)