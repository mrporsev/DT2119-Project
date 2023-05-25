import torch
import torchaudio
from utils_vad import get_speech_timestamps

import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
from preprocessing import Wav2Mel

import numpy as np
from pprint import pprint

print("loading models")
# preprocessing module
sample_rate = 16000 # resample audio to this sample rate
norm_db = -3.0
fft_window_ms = 25.0
fft_hop_ms = 10.0
wav2mel = Wav2Mel(sample_rate, norm_db, fft_window_ms, fft_hop_ms)

# embeding model
dvector = torch.jit.load("dvector-step250000.pt").eval()

# vad model
vad_model = torch.jit.load("silero_vad.jit")


print("loading audio")
# read audio from file
wav_tensor, native_sample_rate = torchaudio.load("gwtwd.wav")
# resample to 16 kHz, apply effects and compute mel spectrogram
wav_tensor, mel_tensor = wav2mel(wav_tensor, native_sample_rate)

print("analyzing audio")
# Use pretrained vad model to find intervals containing speech
speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate, return_seconds=True)
pprint(speech_timestamps)


def get_frames(mel_tensor, block_size):
    return mel_tensor.unfold(0, block_size, block_size).mT


def get_frame_embeddings(mel_frames):
    embeddings = torch.empty(mel_frames.shape[0], 256)
    for frame_idx in range(mel_frames.shape[0]):
        embeddings[frame_idx, :] = dvector.embed_utterance(mel_frames[frame_idx])
    return embeddings


block_size = 50 # block size is the number of MFCC frames to stack togeter for each embeding frame
mel_frames = get_frames(mel_tensor, block_size) 
embed = get_frame_embeddings(mel_frames)

print(mel_frames.shape)
print(embed.shape)
print(embed)

# Unfortunately, it is not easy to verify that the above procedure is correct.
# But the least you can do is having a look at the dimensions and ensuring they look ok.

# The next step is to cluster the embedings 


