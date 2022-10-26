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
import matplotlib.pyplot as plt
import gzip
from scipy.signal import find_peaks,peak_widths



path2folder = "/home/cse_urp_dl2/Documents/hhj/ECG/125/"
df_meta = pd.read_csv("/home/cse_urp_dl2/Documents/hhj/ECG/total1.csv")
ecg_path ="/home/cse_urp_dl2/Documents/hhj/ECG/1010/"
df_train = df_meta[df_meta["AKI"]==0]


def vital_to_npy(i):
    file_id = df_train.iloc[i,1]
    ipath = path2folder + file_id + ".npy"
    npy = np.load(ipath)
    sample_num = int(npy.shape[0]//1250)
    if sample_num == 0: pass
    
    for i in range(sample_num):
        start = i
        sample = npy[start:start+1250]
        sample = sample[~np.isnan(sample)]
        if sample.shape[0] == 0 or sample.shape[0] == 1 : pass
        min_val ,max_val = np.min(sample),np.max(sample)
        norm_sample = (sample -min_val)/(max_val-min_val)
        del sample
        #  find r-peak and slice
        peaks, _ = find_peaks(norm_sample, height=0.9)
        if peaks.shape[0] == 0 or peaks.shape[0] == 1 : pass
        try:
            ecg_len = int(np.median(np.diff(peaks)))
            if ecg_len >= 100 and ecg_len <= 150:
                for peak in peaks:
                    t = peak-int(len/2)
                    seg = norm_sample[t:t+len]
            
                    if (np.sum(np.isnan(seg)) == 0) and (seg.shape[0] != 0):
                        np.save(ecg_path+f"{file_id}_{peak}.npy",seg)
                        
        except:
            pass
        
    del npy
    
for i in tqdm(range(df_train.shape[0])):
    vital_to_npy(i)