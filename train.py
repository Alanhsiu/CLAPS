import os
import time
import json
import torch
import random
import argparse
from model import Model
from accelerate import Accelerator
from dataset.dataset import dataset
from dataset.parser import parse_list
from accelerate.utils import set_seed
from transformers import get_scheduler
from torch.utils.data import DataLoader
set_seed(0)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(a):
    # prepare accelerator
    accelerator = Accelerator(log_with='tensorboard', project_dir=a.project_dir)
    accelerator.init_trackers('logs')
    device = accelerator.device

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
        model.parameters(),
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
    model, optim, sched, train_loader, valid_loader = accelerator.prepare(
        model, optim, sched, train_loader, valid_loader
    )

    # load checkpoint
    print("Loading checkpoint") # lichun added
    if a.ckpt_pth != '':
        accelerator.load_state(a.ckpt_pth)
    step = sched.scheduler.last_epoch+1

    inner_step = 1
    losses = [[] for i in range(a.n_sub+1)]
    print("losses: ", losses) # lichun added
    while step <= a.max_train_steps:
        for batch in train_loader:
            if step > a.max_train_steps:
                break
            start = time.time()
            dids, prompts, wavs = batch
            print("dids: ", dids) # lichun added
            print("prompts: ", prompts) # lichun added
            
            # get the COS similarity of prompt_rep & speech_rep
            print("prompt_rep: ", prompt_rep) # lichun added
            print("speech_rep: ", speech_rep) # lichun added

            loss = model.loss(prompt_rep, speech_rep, dids) 
            losses[dids[0]].append(loss.item())
            losses[-1].append(loss.item())
            accelerator.backward(loss)

            if inner_step%a.gradient_accumulation_steps != 0:
                inner_step += 1
                continue
            inner_step = 1

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
                log['train/lr'] = sched.get_lr()[0]
                accelerator.log(log, step=step)
                losses = [[] for i in range(a.n_sub+1)]
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
    parser.add_argument('-p', '--project_dir', type=str, default='cp_claps',
                        help='directory to save results')
    parser.add_argument('-cp', '--ckpt_pth', type=str, default='',
                        help='path to load a checkpoint')
    # train
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
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
