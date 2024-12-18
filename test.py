import torchaudio
import torch 


print(torch.__version__)
print(torchaudio.__version__)

wavfile = torchaudio.load("/work/b0990106x/trl/CLAPS/test/angry.wav")
print(wavfile)

