import os
import sys
import subprocess
import pylab
import shutil

def execute(cmd):
    """execute cmd on sptk"""
    
    subprocess.call(cmd, shell=True)

def w2r(source, target):
	""" convert wav file (filename) into raw file (target) """
	
	cmd = "wav2raw " + source
	execute(cmd)
	path, ext = os.path.splitext(source)
	current = path + '.raw'
	targetfile = '/'.join(target.split('/')[:-1])
	if current != target:
		try:
			os.remove(target)
		except:
			pass
		shutil.move(current, targetfile)

def r2w(filename, fr):
	cmdr2w = 'raw2wav -s %f %s.raw' % (fr, filename)
	
	execute(cmdr2w)

def mp32wav(source, target):
	cmd = "sox " + source + " " + target
	execute(cmd)
