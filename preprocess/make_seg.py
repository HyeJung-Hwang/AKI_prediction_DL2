import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import io
import pickle
import subprocess
import time
from joblib import Parallel, delayed
import gzip
from scipy.signal import find_peaks,peak_widths
## vital file 경로
path2folder = "E:/01_AKI_tolopogy/01_AKI_tolopogy/hhj/125/"
df_meta = pd.read_csv("C:/Users/CSE-220425/Downloads/total1.csv")

## 병렬
def vital_to_npy(i):
    file_id = df_meta.loc[i,"caseid"]
    ipath = path2folder + file_id + ".npy"
    npy = np.load(ipath)

        # select a 10 s window
    t = int(npy.shape[0]/2)
    sample = npy[t:t+1250]
    del npy
        # normalize
    min_val ,max_val = np.min(sample),np.max(sample)
    norm_sample = (sample -min_val)/(max_val-min_val)
    del sample
        #  find r-peak and slice
    peaks, _ = find_peaks(norm_sample, height=0.9)
    try:
        len = int(np.median(np.diff(peaks)))
        for peak in peaks:
            t = peak-int(len/2)
            seg = norm_sample[t:t+len]
            path = f"C:/Users/CSE-220425/Downloads/{file_id}/"
            os.makedirs(path, exist_ok=True)
            if seg.shape[0] == len:
                np.save(path+f"{peak}.npy",seg)
    except:
        pass

#Parallel(n_jobs = 3 )( delayed(vital_to_npy)(i) for i in tqdm(range(1048)) )

for i in tqdm(range(1048)):
    vital_to_npy(i)
