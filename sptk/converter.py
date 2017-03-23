import struct
import sys
import numpy as np

from sptktools import execute

def mcep2vec(filename):
    mceplist = []
    f = open(filename, "rb")
    while True:
        b = f.read(4)
        if b == b'':
            break
        m = struct.unpack("f", b)[0]
        mceplist.append(m)
    f.close()
    return np.reshape(np.array(mceplist), (-1, 40))

def mfcc2vec(filename):
    mfcclist = []
    f = open(filename, "rb")
    while True:
        b = f.read(4)
        if b == b'':
            break
        m = struct.unpack("f", b)[0]
        mfcclist.append(m)
    f.close()
    return np.reshape(np.array(mfcclist), (-1, 39))

def pitch2vec(filename):
    pitchlist = []
    f = open(filename, "rb")
    while True:
        b = f.read(4)
        if b == b'':
            break
        p = struct.unpack("f", b)[0]
        pitchlist.append(p)
    f.close()
    return np.array(pitchlist)

def sp2vec(filename):
    splist = []
    f = open(filename, "rb")
    while True:
        b = f.read(4)
        if b == b'':
            break
        p = struct.unpack("f", b)[0]
        splist.append(p)
    f.close()
    return np.array(splist).reshape(-1, 257)

def vec2mcep(vec, filename):
    vec = np.ndarray.flatten(vec)
    num = len(vec)
    f = open(filename, 'wb')
    f.write(struct.pack('f' * num, *vec))
    f.close()

def vec2pitch(vec, filename):
    num = len(vec)
    f = open(filename, 'wb')
    f.write(struct.pack('f' * num, *vec))
    f.close()

def synthesize(pfile, mfile, rawfile, wavfile):
	c1 = 'excite -p 80 %s | mlsadf -m 39 -a 0.35 -p 80 %s | x2x +fs > %s' \
	        % (pfile, mfile, rawfile)
	c2 = 'sox -e signed-integer -c 1 -b 16 -r 16000 %s %s' % (rawfile, wavfile)
	
	execute(c1)
	execute(c2)
