import os
import torch
from model import Model
from accelerate import Accelerator
import argparse
import torch.nn.functional as F
import torchaudio
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model(a):
    # Prepare accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Initialize the model
    model = Model(a)
    
    # Load checkpoint
    checkpoint_path = os.path.join(a.ckpt_pth, 'pytorch_model.bin')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.to(device)
    model.eval()
    return model, accelerator

def infer(model, accelerator, prompts, wavs):
    # Load model with weights from checkpoint
    # model, accelerator = load_model(a)

    waveform_tensor = [w.to(accelerator.device) for w in wavs]
    
    # Ensure model is in evaluation mode
    model.eval()

    # Convert prompts and wavs to the format expected by the model
    with torch.no_grad():
        # Encode the text prompts
        prompt_rep = model.encode_text(prompts, accelerator.device)
        
        # Encode the speech inputs
        speech_rep = model.encode_speech(waveform_tensor, accelerator.device)

        # Compute the similarity or other inference operation
        # sim = torch.einsum('i d, j d -> i j', prompt_rep, speech_rep) * model.temp.exp()
        cosine_sim = F.cosine_similarity(prompt_rep, speech_rep)
        # sim returns for example tensor([[232.8942]], device='cuda:0')
        # cosine_sim returns for example tensor([0.9999], device='cuda:0')
        
        return cosine_sim

if __name__ == '__main__':
    a = argparse.Namespace(
        sr=16000,
        text_enc_name='google/flan-t5-large',
        text_enc_dim=1024,
        text_blstm_dim=256,
        speech_enc_name='wavlm',
        speech_enc_dim=768,
        speech_blstm_dim=256,
        rep_dim=512,
        sub_dim=0,
        n_sub=1,  # Number of subspaces, if any
        ckpt_pth='/work/b0990106x/trl/CLAPS/pretrained/7d/cp_claps_blstm_m_50k_v3/cp_0045000',  # Set your checkpoint path
        project_dir='cp_claps'  # Example project directory
    )

    prompt = "A woman shouts in a sad tone."
    # prompt = "Make the audio sound sad."
    wav = torchaudio.load(f"/work/b0990106x/trl/CLAPS/loud_angry.wav")
    real_wav = wav[0]
    if real_wav.shape[0]==1:
        real_wav = real_wav.squeeze(0)
    real_wav = [real_wav]
    
    model, accelerator = load_model(a)
    cosine_sim = infer(model, accelerator, prompt, real_wav)

    print(f"For prompt: {prompt}")
    print(f"Cosine Similarity: {cosine_sim}")
    





    # # Example usage
    # # prompts = ["Play the audio twice."]
    # prompts = [
    #     "Play the audio twice.",
    #     "Mildly decrease the emphasis on the higher frequencies.",
    #     "Considerably abate the bass frequencies.",
    #     "Heighten the chorus effect in the audio by a small amount.",
    #     "Hold off on playing the audio for 1 second.",
    #     "Intensify the sound of the higher frequencies.",
    #     "Give the audio a gradual increase in volume for 5 seconds from the onset.",
    #     "Add a conspicuous chorus effect to the audio.",
    #     "Significantly dampen the vibrations of the high notes.",
    #     "Decrease the pitch of the audio by a moderate amount.",
    #     "Introduce a minor adjustment to the pitch of the audio to make it lower.",
    #     "Enlarge the scope and widen the reach of the sound quality.",
    #     "Amplify the sound to deliver a clearer and brighter rendition.",
    #     "Backtrack the sound.",
    #     "Enlarge the depth of the lower frequencies significantly.",
    #     "Refine the sound to make it more discernible and lively.",
    #     "Reduce the high-end frequencies considerably.",
    #     "Strengthen the chorus effect on the audio.",
    #     "During the introduction of the audio, steadily increase the volume by 5 seconds.",
    #     "Apply a chorus effect that's easily noticeable to the audio."
    # ]

    # # prompts = ["There is even a white row of beehives in the orchard, under the walnut trees."]
    # wavs = []
    # total_cosine_sim = []
    # randomize = False
    # if not randomize: 
    #     for i in range(len(prompts)):
    #         loader = torchaudio.load(f"/work/b0990106x/trl/output/audio_sample/example_save_{i}.wav")
    #         wavs.append(loader)
    #         # print(loader)
    # else:
    #     for i in range(len(prompts)):
    #         wavs_2 = (torch.randn(16000*5), 16000)
    #         wavs.append(wavs_2)
    #         # print(wavs_2)

    # for j, wav in enumerate(wavs):
    #     real_wav = wav[0]
    #     if real_wav.shape[0]==1:
    #         print("real_wav.shape is 1")
    #         real_wav = real_wav.squeeze(0)
    #     real_wav = [real_wav]
    #     cosine_sim = infer(a, prompts[j], real_wav)
    #     total_cosine_sim.append(cosine_sim)

    # for k in range(len(wavs)):
    #     print(f"For prompt: {prompts[k]}")
    #     print(f"and the audio example_save_{k}.wav")
    #     print(f"Cosine Similarity: {total_cosine_sim[k]}")
    #     print("---------------------------------------")

    # print(f"Total Cosine Similarity: {total_cosine_sim}")
    # # wavs = wavs[0]
    # # if wavs.shape[0] == 1:  # Check if there's only one channel
    # #     print("HELLO")
    # #     wavs = wavs.squeeze(0)  # Remove the channel dimension
    # # wavs = [wavs]
    # wavs_2 = [torch.randn(16000*5)]  # Example tensor simulating a 1-second wav file, replace with actual wav data
    # # print(wavs_2)

    # # sim, cosine_sim = infer(a, prompts, wavs)
