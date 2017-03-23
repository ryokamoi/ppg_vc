An implementation of [Phonetic Posteriorgrams for Many-to-One Voice Conversion without Parallel Data Training](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training) for Japanese

# Checklists

- [x] Create a phoneme corpus with Julius
- [ ] AI-ASR
- [ ] PPG -> MCEP

# tree

├── analysis
│   ├── convprep.py
│   ├── synthesize.py
│   └── trainprep.py
├── data*
│   └── convertedfiles
├── lstm
│   └── siasr.py
├── main.sh
├── parser
│   └── phoneme2vec.py
├── phoneme.txt
├── readme.md
├── result
├── segmentation-kit*
│   ├── License.md
│   ├── README.md
│   ├── bin
│   │   ├── julius-4.3.1.exe
│   │   └── yomi2voca.pl
│   ├── models
│   │   ├── hmmdefs_monof_mix16_gid.binhmm
│   │   ├── hmmdefs_ptm_gid.binhmm
│   │   └── logicalTri
│   ├── segment_julius.pl
│   └── wav
│       └── sourcefiles*
├── sptk
│   ├── converter.py
│   ├── extract.py
│   └── sptktools.py
└── target*
    └── targetfiles

* ... not contained in this repository
