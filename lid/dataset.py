from torch.utils.data import DataLoader, Dataset
import numpy as np 
import os 
import torchaudio
from pathlib import Path

class LanguageClassifiDataset(Dataset):
    def __init__(self, mode, root_dir, max_timestep=None):
        self.lang_num = 143
        self.max_timestep = max_timestep
        
        self.lang2idx = {}
        self.idx2lang = {}
        with open(os.path.join(root_dir, "lang2idx.txt")) as fp:
            for x in fp:
                idx, lang = x.strip().split(",")
                self.lang2idx[lang] = int(idx)
                self.idx2lang[int(idx)] = lang

        self.dataset = []
        self.label = []
        with open(os.path.join(root_dir, f"{mode}.csv")) as fp:
            for x in fp:
                lang, pth = x.strip().split(",")
                self.dataset.append(pth)
                self.label.append(self.lang2idx[lang])

        print(f"[LanguageIdentificationDataset] Obtain {mode} data of length {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]
        path = self.dataset[idx]
        
        def path2name(path):
            return Path("-".join((Path(path).parts)[-4:])).stem

        return wav.numpy(), self.label[idx], path2name(path)
        
    def collate_fn(self, samples):
        return zip(*samples)
