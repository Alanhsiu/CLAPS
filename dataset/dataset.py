import torch
import random
from librosa import load
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, flist, weights, a):
        self.flist = flist
        self.weights = weights
        self.batch_size = a.batch_size
        self.sr = a.sr
        self.max_l = a.max_l

    def __getitem__(self, index):
        flist = random.choices(self.flist, weights=self.weights, k=1)[0]
        batch = []
        sel_lbls = []
        while len(batch) < self.batch_size:
            sample = random.choice(flist)
            if sample[1] is None or sample[1] not in sel_lbls:
                batch.append(sample)
                sel_lbls.append(sample[1])
        dids = []
        prompts = []
        wavs = []
        for did, _, prompt, wav in batch:
            dids.append(did)
            if type(prompt) is list:
                prompt = random.choice(prompt)
            prompts.append(prompt)
            wav = load(wav, sr=self.sr)[0]
            if len(wav) > self.max_l:
                s = random.randint(0, len(wav)-self.max_l)
                wav = wav[s:s+self.max_l]
            wavs.append(torch.Tensor(wav))
        return dids, prompts, wavs # 要餵給他ㄉ東西

    def __len__(self):
        return sum([len(x) for x in self.flist])
