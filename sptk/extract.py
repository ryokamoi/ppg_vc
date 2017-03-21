from sptktools import execute

def ext_mcep(rawfile, mcepfile):
	mcepcmd = 'x2x +sf < %s | frame -l 400 -p 80 | window -l 400 -L 512 -w 1 | mcep -l 512 -m 39 -a 0.35 -e 0.00001 > %s' % (rawfile, mcepfile)
	
	execute(mcepcmd)

def ext_mfcc(rawfile, mfccfile):
	mfcccmd = 'x2x +sf < %s | frame -l 400 -p 80 | mfcc -l 400 -m 39 > %s' % (rawfile, mfccfile)
	
	execute(mfcccmd)

def ext_pitch(rawfile, pitchfile):
	f0cmd = 'x2x +sf %s | pitch -a 1 -p 80 -s 16.0 > %s' \
	                                       % (rawfile, pitchfile)
	execute(f0cmd)

def ext_f0(rawfile, pitchfile):
	f0cmd = 'x2x +sf %s | pitch -a 1 -p 80 -s 16.0 -o 1 > %s' \
	                                       % (rawfile, pitchfile)
	execute(f0cmd)
	
def ext_logf0(rawfile, pitchfile):
	f0cmd = 'x2x +sf %s | pitch -a 1 -p 80 -s 16.0 -o 2  > %s' \
	                                       % (rawfile, pitchfile)
	execute(f0cmd)

def mcep2sp(mfile, spfile):
	cmd = 'mgc2sp -m 39 -a 0.42 -l 512 -o 2 %s > %s' \
	                                       % (mfile, spfile)
	execute(cmd)
