#!/bin/sh

python analysis/convprep.py
python analysis/synthesize.py
python parser/phoneme2vec.py
python parser/sizeunification.py
python lstm/siasr.py