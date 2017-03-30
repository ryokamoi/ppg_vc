import os
import fnmatch
import re
import wave
from enum import Enum

datapath = 'pasd/'
savepath = 'segmentation-kit/wav/'

class Univ(Enum):
    chi = 0
    kyo = 1
    osa = 2
    tsu = 3
    uec = 4
    was = 5
    kan = 6
    tit = 7
    siz = 8
    tok = 9
    tut = 10

txtspeaker = [('G:', 'F:'), ('S:', 'U:'), ('U:', 'S:'), ('A:', 'B:'), ('A:', 'B:'), \
                      ('A:', 'B:'), ('A:', 'B:'), ('A:', 'B:'), ('A:', 'B:'), ('A:', 'B:'), ('A:', '')]
wavspeaker = [('l', 'r'), ('l', 'r'), ('l', 'r'), ('l', 'r'), ('a', 'b'), \
                        ('l', 'r'), ('l', 'r'), ('l', 'r'), ('l', 'r'), ('l', 'r'), ('l', '')]

def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

splist1 = ['\n', '\r']
sublist = [r'<.*?>', r'＜.*?＞', r'{.*?}', r'#.*?#', r'\+.*?\+', r'tut[0-9]+']
splist2 = ['、', '。', '-', '．', '{comma}', '*pause*', '*silence*', '{period}', \
             '{quest}','pencil', '#', 'h', '[', ']', '(', ')', '（', '）', '［', '］', '｛', '｝', '/cg/', '/lg/', \
             '/ls/', '/', '+', '？']
def kan2txt(filename, univ):
    global txtlist
    
    root, ext = os.path.splitext(filename)
    root = root.split('/')[-1]
    savename = [savepath+root+wavspeaker[univ.value][i]+'.txt' for i in range(2)]
    
    savetxt  = [[] for i in range(2)]
    f = open(filename, "rb")
    for l in f.readlines():
        line = l.decode('sjis')
        if line[:2] in txtspeaker[univ.value] or line[:2] == '  ':
            if line[:2] in txtspeaker[univ.value]:
                speaker = line[:2]
            line = re.sub(r'\s+', 'sp', line)
            line = line.replace(line[:2], 'sp')
            for seg in splist1:
                line = line.replace(seg, 'sp')
            for seg in sublist:
                line = re.sub(seg, 'sp', line)
            for seg in splist2:
                line = line.replace(seg, 'sp')
            savetxt[txtspeaker[univ.value].index(speaker)].append(line)
    f.close()
    
    for i in range(2):
        if wavspeaker[univ.value][i]:
            txtlist.append(savename[i])
            f = open(savename[i], 'w')
            savetxt[i] = re.sub(r'sp(sp)*', ' sp ', ''.join(savetxt[i]))
            f.write(savetxt[i])
            f.close()

def wavconnecter(inputs, output):
    fps = [wave.open(f, 'r') for f in inputs]
    fpw = wave.open(output, 'w')
    
    fpw.setnchannels(fps[0].getnchannels())
    fpw.setsampwidth(fps[0].getsampwidth())
    fpw.setframerate(fps[0].getframerate())
    
    for fp in fps:
        fpw.writeframes(fp.readframes(fp.getnframes()))
        fp.close()
    fpw.close()

if __name__ == '__main__':
    kanfiles = [[] for i in list(Univ)]
    wavfiles = [[] for i in list(Univ)]
    for filename in fild_all_files(datapath):
        if fnmatch.fnmatch(filename, '*.kan'):
            univ = filename.split('/')[-3]
            kanfiles[Univ[univ].value].append(filename)
        elif fnmatch.fnmatch(filename, '*.wav'):
            if filename.split('/')[-2] != 'wav_dlg':
                univ = filename.split('/')[-3]
                wavfiles[Univ[univ].value].append(filename)
    
    txtlist = []
    for univ in list(Univ):
        for kanname in kanfiles[univ.value]:
            kan2txt(kanname, univ)
    
    wavlist = [txtname[:-3]+'wav' for txtname in txtlist]
    for savewav in wavlist:
        print(savewav)
        inputlist = []
        saveroot, saveext = os.path.splitext(savewav)
        saveroot = saveroot.split('/')[-1]
        for univ in list(Univ):
            for wavname in wavfiles[univ.value]:
                inputroot, inputext = os.path.splitext(wavname)
                inputroot = inputroot.split('/')[-1]
                if saveroot == inputroot[:len(saveroot)]:
                    inputlist.append(wavname)
        wavconnecter(inputlist, savewav)
