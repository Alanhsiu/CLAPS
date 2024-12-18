import os
import time
import json
import torch
import random
import argparse
from model import Model
from librosa import load
import torch.nn.functional as F
from accelerate import Accelerator
from dataset.dataset import dataset
from torch.utils.data import Dataset
from dataset.parser import parse_list
from accelerate.utils import set_seed
from transformers import get_scheduler
from torch.utils.data import DataLoader
set_seed(0)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class hf_dataset(Dataset):
    def __init__(self, data_dir, csv, sr, max_l):
        with open(csv, 'r') as r:
            self.flist = [x.strip().split(',') for x in r.readlines()]
        self.flist = [[
            x[0], os.path.join(data_dir, x[1]),
            os.path.join(data_dir, x[2]), int(x[3])
        ] for x in self.flist]
        self.sr = sr
        self.max_l = max_l

    def __getitem__(self, index):
        prompt, uttr_a, uttr_b, lab = self.flist[index]
        uttr_a = self.load(uttr_a)
        uttr_b = self.load(uttr_b)
        return prompt, uttr_a, uttr_b, torch.LongTensor([lab])

    def load(self, wav):
        wav = load(wav, sr=self.sr)[0]
        if len(wav) > self.max_l:
            s = random.randint(0, len(wav)-self.max_l)
            wav = wav[s:s+self.max_l]
        return torch.Tensor(wav)

    def __len__(self):
        return len(self.flist)


class HF_Loss_Fn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, prompt_reps, uttr_a_reps, uttr_b_reps, labs):
        uttr_a_sims = F.cosine_similarity(prompt_reps, uttr_a_reps)
        uttr_b_sims = F.cosine_similarity(prompt_reps, uttr_b_reps)
        acc = labs == (uttr_b_sims > uttr_a_sims)
        acc = sum(acc)/len(acc)
        loss = -torch.log(torch.sigmoid((2*labs-1)*(uttr_b_sims-uttr_a_sims)*self.temp.exp()))
        return loss.mean(), acc


def train(a):
    # prepare accelerator
    accelerator = Accelerator(log_with='tensorboard', project_dir=a.project_dir)
    accelerator.init_trackers('logs')
    device = accelerator.device

    # prepare hf dataloaders, loss_fn
    hf_train_dataset = hf_dataset(a.data_dir, a.hf_train_csv, a.sr, a.max_l)
    hf_train_loader = DataLoader(
        hf_train_dataset,
        num_workers=0,
        shuffle=True,
        batch_size=a.hf_batch_size,
        pin_memory=True,
        collate_fn=lambda x: [e for e in zip(*x)]
    )
    hf_valid_dataset = hf_dataset(a.data_dir, a.hf_valid_csv, a.sr, a.max_l)
    hf_valid_loader = DataLoader(
        hf_valid_dataset,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        collate_fn=lambda x: [e for e in zip(*x)]
    )
    hf_loss_fn = HF_Loss_Fn()

    # prepare dataloaders, model, optimizer, scheduler
    train_flist = parse_list(a.data_dir, 'train')
    weights = [len(f) for f in train_flist]
    train_dataset = dataset(train_flist, weights, a)
    train_loader = DataLoader(
        train_dataset,
        num_workers=a.n_workers,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )
    valid_flist = parse_list(a.data_dir, 'valid')
    valid_dataset = dataset(valid_flist, weights, a)
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=a.n_workers,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )
    a.n_sub = len(train_flist)
    model = Model(a)
    optim = torch.optim.Adam(
        list(model.parameters())+list(hf_loss_fn.parameters()),
        lr=a.lr,
        betas=(a.beta1, a.beta2),
        weight_decay=a.weight_decay
    )
    sched = get_scheduler(
        name=a.lr_scheduler_type,
        optimizer=optim,
        num_warmup_steps=a.num_warmup_steps,
        num_training_steps=a.max_train_steps
    )
    with open(os.path.join(a.project_dir, 'config.json'), 'w') as f:
        json.dump(a.__dict__, f, indent=4)

    # pass to accelerator
    model, hf_loss_fn, optim, sched, hf_train_loader, hf_valid_loader, train_loader, valid_loader = \
    accelerator.prepare(
        model, hf_loss_fn, optim, sched, hf_train_loader, hf_valid_loader, train_loader, valid_loader
    )

    # load checkpoint
    if a.ckpt_pth != '':
        accelerator.load_state(a.ckpt_pth)
    step = sched.scheduler.last_epoch+1

    inner_step = 1
    losses = [[] for i in range(a.n_sub+1)]
    hf_losses, hf_accs = [], []
    while step <= a.max_train_steps:
        for batch in train_loader:
            if step > a.max_train_steps:
                break
            start = time.time()
            dids, prompts, wavs = batch
            prompt_rep = model.encode_text(prompts, device)
            speech_rep = model.encode_speech(wavs, device)

            loss = model.loss(prompt_rep, speech_rep, dids)
            losses[dids[0]].append(loss.item())
            losses[-1].append(loss.item())
            accelerator.backward(loss)

            if inner_step%a.gradient_accumulation_steps != 0:
                inner_step += 1
                continue
            inner_step = 1

            if step%a.hf_steps == 0 and step > a.hf_after_steps:
                prompts, uttr_as, uttr_bs, labs = next(iter(hf_train_loader))
                labs = torch.cat(labs)
                prompt_reps = model.encode_text(prompts, device)
                uttr_a_reps = model.encode_speech(uttr_as, device)
                uttr_b_reps = model.encode_speech(uttr_bs, device)
                hf_loss, hf_acc = hf_loss_fn(prompt_reps, uttr_a_reps, uttr_b_reps, labs)
                hf_losses.append(hf_loss.item())
                hf_accs.append(hf_acc.item())
                accelerator.backward(hf_loss)

            optim.step()
            sched.step()
            optim.zero_grad()
            d = time.time()-start
            accelerator.wait_for_everyone()

            if step%a.print_steps == 0:
                accelerator.print(
                    'Steps : {:d}, Loss : {:4.3f}, s/b : {:4.3f}'.format(
                        step, loss.item(), d))
            if step%a.log_steps == 0:
                log = {}
                for i in range(a.n_sub):
                    if len(losses[i]) > 0:
                        log['train/loss_{:1d}'.format(i)] = sum(losses[i])/len(losses[i])
                log['train/loss'] = sum(losses[-1])/len(losses[-1])
                if len(hf_losses) > 0:
                    log['train/hf_loss'] = sum(hf_losses)/len(hf_losses)
                    log['train/hf_acc'] = sum(hf_accs)/len(hf_accs)
                log['train/lr'] = sched.get_lr()[0]
                accelerator.log(log, step=step)
                losses = [[] for i in range(a.n_sub+1)]
                hf_losses = []
                hf_accs = []
            if step%a.save_steps == 0:
                accelerator.save_state('{}/cp_{:07d}'.format(a.project_dir, step))
            if step%a.valid_steps == 0:
                model.eval()
                valid_losses = [[] for i in range(a.n_sub+1)]
                for batch in valid_loader:
                    dids, prompts, wavs = batch
                    with torch.no_grad():
                        prompt_rep = model.encode_text(prompts, device)
                        speech_rep = model.encode_speech(wavs, device)
                        loss = model.loss(prompt_rep, speech_rep, dids)
                        valid_losses[dids[0]].append(loss.item())
                        valid_losses[-1].append(loss.item())
                for i in range(a.n_sub):
                    if len(valid_losses[i]) > 0:
                        log['valid/loss_{:1d}'.format(i)] = sum(valid_losses[i])/len(valid_losses[i])
                log['valid/loss'] = sum(valid_losses[-1])/len(valid_losses[-1])
                
                
                hf_valid_losses, hf_valid_accs = [], []
                for batch in hf_valid_loader:
                    prompts, uttr_as, uttr_bs, labs = batch
                    labs = torch.cat(labs)
                    with torch.no_grad():
                        prompt_reps = model.encode_text(prompts, device)
                        uttr_a_reps = model.encode_speech(uttr_as, device)
                        uttr_b_reps = model.encode_speech(uttr_bs, device)
                        hf_loss, hf_acc = hf_loss_fn(prompt_reps, uttr_a_reps, uttr_b_reps, labs)
                        hf_valid_losses.append(hf_loss.item())
                        hf_valid_accs.append(hf_acc.item())
                log['valid/hf_loss'] = sum(hf_valid_losses)/len(hf_valid_losses)
                log['valid/hf_acc'] = sum(hf_valid_accs)/len(hf_valid_accs)

                accelerator.log(log, step=step)
                model.train()

            step += 1
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # env 
    parser.add_argument('-c', '--config', type=str, default='',
                        help='config for training, which will override other arguments')
    parser.add_argument('-d', '--data_dir', type=str, default='data',
                        help='directory to load data')
    parser.add_argument('--hf_train_csv', type=str,
                        default='data/4000v1-v3_train.csv')
    parser.add_argument('--hf_valid_csv', type=str,
                        default='data/4000v1-v3_valid.csv')
    parser.add_argument('-p', '--project_dir', type=str, default='cp_claps',
                        help='directory to save results')
    parser.add_argument('-cp', '--ckpt_pth', type=str, default='',
                        help='path to load a checkpoint')
    # train
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--hf_batch_size', type=int, default=1,
                        help='batch size for hf')
    parser.add_argument('--hf_steps', type=int, default=4,
                        help='steps for hf')
    parser.add_argument('--hf_after_steps', type=int, default=0,
                        help='start step for hf')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        help='learning rate scheduler type')
    parser.add_argument('--num_warmup_steps', type=int, default=5000,
                        help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='steps for gradient accumulation')
    parser.add_argument('--max_train_steps', type=int, default=50e3,
                        help='total number of training steps')
    parser.add_argument('--print_steps', type=int, default=5,
                        help='steps to print during training')
    parser.add_argument('--log_steps', type=int, default=100,
                        help='steps to log during training')
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='steps to save during training')
    parser.add_argument('--valid_steps', type=int, default=5000,
                        help='steps to validate during training')
    # optimizer
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 parameter for optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 parameter for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for optimizer')
    # model
    parser.add_argument('--sr', type=int, default=16000,
                        help='sampling rate of speech encoder')
    parser.add_argument('--max_l', type=int, default=160000,
                        help='max length of speech')
    parser.add_argument('--text_enc_name', type=str, default='google/flan-t5-large',
                        help='name of text encoder')
    parser.add_argument('--text_enc_dim', type=int, default=1024,
                        help='dimension of text encoder')
    parser.add_argument('--text_blstm_dim', type=int, default=256,
                        help='dimension of text blstm')
    parser.add_argument('--speech_enc_name', type=str, default='wavlm',
                        help='name of speech encoder')
    parser.add_argument('--speech_enc_dim', type=int, default=768,
                        help='dimension of speech encoder')
    parser.add_argument('--speech_blstm_dim', type=int, default=256,
                        help='dimension of speech blstm')
    parser.add_argument('--rep_dim', type=int, default=512,
                        help='dimension of shared space representation')
    parser.add_argument('--sub_dim', type=int, default=0,
                        help='dimension of shared subspace representation')

    a = parser.parse_args()
    if a.config != '':
        with open(a.config, 'r') as f:
            a.__dict__ = json.load(f)

    train(a)
