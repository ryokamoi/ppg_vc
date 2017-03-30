import os
import sys

import numpy as np

ppgname = 'resultppg.npy'
savepath = ''

PHONEME = 36

f = open('phoneme.txt')
phonemelist = []
for phoneme in f.readlines():
    phonemelist.append(phoneme.replace('\n', ''))
f.close()
phonemedic = {phonemelist.index(p):p for p in phonemelist}

root, ext = os.path.splitext(ppgname)
root = root.split('/')[-1]
ppg = np.load(ppgname)
if len(np.shape(ppg)) > 2:
    ppg = ppg.reshape(-1, PHONEME)
maxphoneme = [None for i in range(np.shape(ppg)[0])]
for i in range(np.shape(ppg)[0]):
    maxphoneme[i] = phonemelist[np.argmax(ppg[i])] + '\n'

f = open(savepath+root+'.txt', 'w')
f.writelines(maxphoneme)
f.close()
