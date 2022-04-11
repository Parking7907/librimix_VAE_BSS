import sys
sys.path.append('./')
import argparse
import pdb
import argparse
import yaml 
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from tqdm import tqdm
#from train import Trainer
#from test import total_test
#from setup import setup_solver

import torch
import torchaudio
import torchaudio.functional
import logging
from glob import glob
data_path = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/min/train-360/*/*'
#/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1/
music_list = glob(data_path)
data_path2 = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/min/test/*/*'
music_list2 = glob(data_path2)
#pdb.set_trace()
print("music list :", len(music_list), len(music_list2))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 기본적으로 44.1kHz, => Win_len = 40ms, n_fft = win_len보다 크게. hop_len = 10ms.. 굳이?
win_len = 1024
hop_length = 256
n_fft = 1024
sampling = 16000
#fs = 4
#duration_len = 3
#length = sampling * duration_len
save_path = '/home/data/jinyoung/source_separation/Libri2Mix/wav16k/min_spectrogram_512/'
window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)


i= 0 
for music_name in music_list:
    music_n = music_name.split('/')[-1]
    dir_n = music_name.split('/')[-2]
    music_n = music_n.split('.wav')[0]
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 1, 235200
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False, return_complex = True)#, return_complex = False)
    #vocal_spectrogram = 1 X 1025 X 460
    vocal_real_0 = vocal_spectrogram[0, :256, :] # 513, 390
    
    '''
    input_real_0 = input_spectrogram[0,:,:,0] # B, 2, 1025, 259
    input_imag_0 = input_spectrogram[0,:,:,1] # B, 2, 1025, 259
    input_real_1 = input_spectrogram[1,:,:,0] # B, 2, 1025, 259?
    input_imag_1 = input_spectrogram[1,:,:,1] # B, 2, 1025, 259
    '''
    vocal_real_0 = vocal_real_0.numpy()
    os.makedirs(save_path + 'train-360/' + dir_n + '/', exist_ok=True)
    out_dir = save_path + 'train-360/' + dir_n + '/' + music_n + '.npy'
    #print("outdir =", out_dir)
    #pdb.set_trace()
    np.save(out_dir, vocal_real_0)
    if i % 1000 ==0 :
        print("Done %i/%i"%(i, len(music_list)))
    i+=1

for music_name in music_list2:
    music_n = music_name.split('/')[-1]
    dir_n = music_name.split('/')[-2]
    music_n = music_n.split('.wav')[0]
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 1, 235200
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False, return_complex = True)#, return_complex = False)
    #vocal_spectrogram = 1 X 1025 X 460
    vocal_real_0 = vocal_spectrogram[0, :256, :] # 513, 390
    
    '''
    input_real_0 = input_spectrogram[0,:,:,0] # B, 2, 1025, 259
    input_imag_0 = input_spectrogram[0,:,:,1] # B, 2, 1025, 259
    input_real_1 = input_spectrogram[1,:,:,0] # B, 2, 1025, 259?
    input_imag_1 = input_spectrogram[1,:,:,1] # B, 2, 1025, 259
    '''
    vocal_real_0 = vocal_real_0.numpy()
    os.makedirs(save_path + 'test/' + dir_n + '/', exist_ok=True)
    out_dir = save_path + 'test/' + dir_n + '/' + music_n + '.npy'
    #print("outdir =", out_dir)
    #pdb.set_trace()
    np.save(out_dir, vocal_real_0)
    if i % 1000 ==0 :
        print("Done %i/%i"%(i, len(music_list)))
    i+=1