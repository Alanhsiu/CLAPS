import os
import csv
import json
import random


def parse_list(data_dir, split):
    promptttsr_list = parse_promptttsr(data_dir, split)
    promptttss_list = parse_promptttss(data_dir, split)
    emovdb_list = parse_emovdb(data_dir, split)
    esd_list = parse_esd(data_dir, split)
    cremad_list = parse_cremad(data_dir, split)
    vctk_list = parse_vctk(data_dir, split)
    audiocaps_list = parse_audiocaps(data_dir, split)
    random.shuffle(promptttsr_list)
    random.shuffle(promptttss_list)
    random.shuffle(emovdb_list)
    random.shuffle(esd_list)
    random.shuffle(cremad_list)
    random.shuffle(vctk_list)
    random.shuffle(audiocaps_list)
    return [
        promptttsr_list, promptttss_list, emovdb_list,
        esd_list, cremad_list, vctk_list, audiocaps_list
    ]


# 0 - PromptTTS-R
def parse_promptttsr(data_dir, split):
    base_dir = os.path.join(data_dir, 'PromptTTS-R')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l for l in csv.reader(r, delimiter=',', quotechar='"')]
    for l in lines[1:]:
        wav_path = os.path.join(base_dir, 'wavs', l[0]+'.wav')
        flist.append([0, '-'.join(l[2:6]), l[6], wav_path])
    return flist


# 1 - PromptTTS-S
def parse_promptttss(data_dir, split):
    base_dir = os.path.join(data_dir, 'PromptTTS-S')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l for l in csv.reader(r, delimiter=',', quotechar='"')]
    for l in lines[1:]:
        wav_path = os.path.join(base_dir, 'wavs', l[0])
        flist.append([1, '-'.join(l[2:7]), l[7], wav_path])
    return flist


# 2 - EmoV-DB
def parse_emovdb(data_dir, split):
    base_dir = os.path.join(data_dir, 'EmoV-DB')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l.strip() for l in r.readlines()]
    with open(os.path.join(base_dir, 'prompts.json'), 'r') as r:
        prompts = json.load(r)
    for l in lines:
        spk2gen = {'bea': 'F', 'jenie': 'F', 'josh': 'M', 'sam': 'M'}
        spk = l.split('/')[0]
        gen = spk2gen[spk]
        emo = l.split('/')[1].split('_')[0].lower()
        prompt = prompts[gen][emo]
        wav_path = os.path.join(base_dir, 'wavs', l)
        flist.append([2, gen+'-'+emo, prompt, wav_path])
    return flist


# 3 - ESD
def parse_esd(data_dir, split):
    base_dir = os.path.join(data_dir, 'ESD')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l.strip() for l in r.readlines()]
    with open(os.path.join(base_dir, 'prompts.json'), 'r') as r:
        prompts = json.load(r)
    for l in lines:
        spk2gen = {
            '0011': 'M', '0012':'M', '0013': 'M', '0014':'M', '0015': 'F',
            '0016': 'F', '0017':'F', '0018': 'F', '0019':'F', '0020': 'M'
        }
        spk = l.split('/')[0]
        gen = spk2gen[spk]
        emo = l.split('/')[1].split('_')[0].lower()
        prompt = prompts[gen][emo]
        wav_path = os.path.join(base_dir, 'wavs', l)
        flist.append([3, gen+'-'+emo, prompt, wav_path])
    return flist


# 4 - CREMA-D
def parse_cremad(data_dir, split):
    base_dir = os.path.join(data_dir, 'CREMA-D')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l.strip().split(',') for l in r.readlines()[1:]]
    with open(os.path.join(base_dir, 'prompts.json'), 'r') as r:
        prompts = json.load(r)
    for l in lines:
        wav_path = os.path.join(base_dir, 'wavs', l[0])
        l[4] = 'XX' if l[4] == 'X' else l[4]
        prompt = prompts[l[2]][l[3]][l[4]]
        flist.append([4, '-'.join(l[2:5]), prompt, wav_path])
    return flist


# 5 - VCTK
def parse_vctk(data_dir, split):
    base_dir = os.path.join(data_dir, 'VCTK')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l.strip().split(',') for l in r.readlines()[1:]]
    with open(os.path.join(base_dir, 'prompts.json'), 'r') as r:
        prompts = json.load(r)
    for l in lines:
        wav_path = os.path.join(base_dir, 'wavs', l[0])
        prompt = prompts[l[2]][l[3]]
        flist.append([5, '-'.join(l[2:4]), prompt, wav_path])
    return flist


# 6 - AudioCaps
def parse_audiocaps(data_dir, split):
    base_dir = os.path.join(data_dir, 'AudioCaps')
    flist = []
    with open(os.path.join(base_dir, split+'.csv'), 'r') as r:
        lines = [l for l in csv.reader(r, delimiter=',', quotechar='"')]
    for l in lines:
        wav_path = os.path.join(base_dir, 'wavs', l[0])
        flist.append([6, None, l[1], wav_path])
    return flist
