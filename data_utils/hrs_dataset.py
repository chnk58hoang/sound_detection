import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import os
from asteroid.data.fuss_dataset import FUSSDataset


class HSRDataset(Dataset):
    def __init__(self,
                 file_list_path: str,
                 mixture_file_dir: str,
                 source_file_dir: str,
                 bg_file_dir: str,
                 sample_rate: int = 16000,
                 return_bg=False):
        """_summary_

        Args:
            file_list_path (_type_): path to the csv file
            return_bg (bool, optional)
        """
        self.mixture_file_dir = mixture_file_dir
        self.source_file_dir = source_file_dir
        self.bg_file_dir = bg_file_dir
        self.return_bg = return_bg
        self.max_n_fg = 1
        self.n_src = self.max_n_fg
        self.sample_rate = 16000
        self.num_samples = self.sample_rate * 10
        self.fg_name = ['machine']
        names = ["mix", "bg"] + self.fg_name
        self.mix_df = pd.read_csv(file_list_path, sep='\t')
    
    def __len__(self):
        return len(self.mix_df)
    
    def __getitem__(self, idx):
         # Each line has absolute to mixture, background and foregrounds
        line = self.mix_df.iloc[idx]
        mixsound_file_path = os.path.join(self.mixture_file_dir, line["mix"]) 
        mix = sf.read(mixsound_file_path, dtype="float32")[0]
        machine_sound_file_path = os.path.join(self.source_file_dir, line["machine"])
        source = sf.read(machine_sound_file_path, dtype="float32")[0]
        source = torch.from_numpy(source).unsqueeze(0)
        if self.return_bg:
            bg_file_path = os.path.join(self.bg_file_dir, line["bg"])
            bg = sf.read(bg_file_path, dtype="float32")[0]
            return torch.from_numpy(mix), source, torch.from_numpy(bg)
        return torch.from_numpy(mix), source