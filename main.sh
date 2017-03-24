#!/bin/sh

cd segmentation-kit
perl segment_julius.pl
cd ../
python analysis/siasrprep.py
python analysis/convprep.py
python analysis/synthesize.py
python parser/phoneme2vec.py
python parser/siasrunification.py
python lstm/siasr.py
python lstm/conv2ppg.py
python parser/ppgunification.py
python lstm/siasr.py
