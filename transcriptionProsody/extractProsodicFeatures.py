import os
import time

import numpy as np
import pandas as pd
import sys 

import audb
import audiofile
import opensmile

IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

signal, sampling_rate = audiofile.read(
    IN_PATH,
    always_2d=True,
)

#note increasing num_workers doesn't seem to help much here 
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors, 
    num_workers=1, 
    multiprocessing=True
)

df = smile.process_signal(
    signal,
    sampling_rate
)
df.to_csv(OUT_PATH + "LowLevel.csv")

"""
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    num_workers=2, 
    multiprocessing=True
) 

oldTime = time.time()
df = smile.process_signal(
    signal,
    sampling_rate
)
df.to_csv(OUT_PATH + "Functionals.csv")
print(f"{time.time() - oldTime}")
"""

