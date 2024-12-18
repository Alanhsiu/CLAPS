import torch
from torch import nn, einsum
from s3prl.nn import S3PRLUpstream
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers import T5EncoderModel
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import logging


# logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)


class Model(nn.Module):
    def __init__(self, a):
        super().__init__()
        # text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(a.text_enc_name)
        self.text_encoder = T5EncoderModel.from_pretrained(a.text_enc_name)
        self.text_blstm = nn.LSTM(
            a.text_enc_dim, a.text_blstm_dim, bidirectional=True, batch_first=True)
        self.text_rep_lin = nn.Linear(a.text_blstm_dim*2, a.rep_dim)

        # speech encoder
        self.speech_encoder = S3PRLUpstream(a.speech_enc_name)
        self.speech_encoder.upstream.model.feature_grad_mult = 1
        self.weight = nn.Parameter(torch.randn(
            (self.speech_encoder._num_layers, 1, 1, 1)))
        self.speech_blstm = nn.LSTM(
            a.speech_enc_dim, a.speech_blstm_dim, bidirectional=True, batch_first=True)
        self.speech_rep_lin = nn.Linear(a.speech_blstm_dim*2, a.rep_dim)

        # subspace linear
        self.sub_lin = None
        if a.sub_dim > 0:
            self.sub_lin = nn.ModuleList()
            for i in range(a.n_sub):
                self.sub_lin.append(nn.Linear(a.rep_dim, a.sub_dim))

        self.temp = nn.Parameter(torch.tensor(1.))

    def encode_text(self, prompts, device):
        # encode prompts using pre-trained LLM, (B, T, text_enc_dim)
        prompts = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length,
                                 padding=True, truncation=True, return_tensors='pt')
        input_ids = prompts.input_ids.to(device)
        attention_mask = prompts.attention_mask.to(device)
        prompt_rep_len = prompts.attention_mask.sum(-1).cpu()
        with torch.no_grad():
            self.text_encoder.eval()
            prompt_rep = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask)[0]

        # blstm, (B, T, text_enc_dim) -> (B, T, text_blstm_dim*2)
        packed_prompt_rep = pack_padded_sequence(
            prompt_rep, prompt_rep_len, batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.text_blstm(packed_prompt_rep)
        prompt_rep = pad_packed_sequence(out, batch_first=True)[0]

        # average pooling (B, T, text_blstm_dim*2) -> (B, text_blstm_dim*2)
        prompt_rep = prompt_rep.sum(1)/prompt_rep_len.unsqueeze(1).to(device)

        # linear, (B, text_blstm_dim*2) -> (B, rep_dim)
        prompt_rep = self.text_rep_lin(prompt_rep)
        return prompt_rep

    def encode_speech(self, wavs, device, with_grad=False):
        # encode wavs using pre-trained LLM, (B, T, speech_enc_dim)
        wavs_len = torch.LongTensor([len(x) for x in wavs])
        wavs = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        if with_grad:
            self.speech_encoder.train()
            speech_rep, speech_rep_len = self.speech_encoder(wavs, wavs_len)
        else:
            with torch.no_grad():
                self.speech_encoder.eval()
                speech_rep, speech_rep_len = self.speech_encoder(wavs, wavs_len)
        speech_rep = torch.stack(speech_rep, 0)
        speech_rep_len = speech_rep_len[0]
        speech_rep = (self.weight*speech_rep).sum(0)

        # blstm, (B, T, speech_enc_dim) -> (B, T, speech_blstm_dim*2)
        packed_speech_rep = pack_padded_sequence(
            speech_rep, speech_rep_len, batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.speech_blstm(packed_speech_rep)
        speech_rep = pad_packed_sequence(out, batch_first=True)[0]

        # average pooling (B, T, speech_blstm_dim*2) -> (B, speech_blstm_dim*2)
        speech_rep = speech_rep.sum(1)/speech_rep_len.unsqueeze(1).to(device)

        # linear, (B, speech_blstm_dim*2) -> (B, rep_dim)
        speech_rep = self.speech_rep_lin(speech_rep)
        return speech_rep

    def loss(self, prompt_rep, speech_rep, did):
        if self.sub_lin is not None:
            # project to different subspaces
            # support samples from different datasets in a batch
            # (B, rep_dim) -> (B, sub_dim)
            sub_prompt_rep = []
            sub_speech_rep = []
            for i, rng in enumerate(self.did2rng(did)):
                s, e = rng
                sub_prompt_rep.append(self.sub_lin[i](prompt_rep[s:e]))
                sub_speech_rep.append(self.sub_lin[i](speech_rep[s:e]))
            prompt_rep = torch.cat(sub_prompt_rep, 0)
            speech_rep = torch.cat(sub_speech_rep, 0)

        # calculate CLIP loss
        # ref: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py#L345C9-L347C85
        prompt_rep = F.normalize(prompt_rep, p=2, dim=-1)
        speech_rep = F.normalize(speech_rep, p=2, dim=-1)
        sim = einsum('i d, j d -> i j', prompt_rep, speech_rep)*self.temp.exp()
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (F.cross_entropy(sim, labels)+F.cross_entropy(sim.t(), labels))/2
        return loss

    def did2rng(self, did):
        rngs = []
        s = 0
        e = 1
        prev_did = did[0]
        for i in range(len(did)):
            if prev_did != did[i]:
                e = i
                rngs.append([s, e])
                s = e
            if i == len(did)-1:
                rngs.append([s, len(did)])
            prev_did = did[i]
        assert rngs[0][0] == 0
        assert rngs[-1][1] == len(did)
        for i in range(len(rngs)-1):
            assert rngs[i][1] == rngs[i+1][0]
        return rngs
