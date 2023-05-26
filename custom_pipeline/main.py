from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch
import torchaudio
from utils_vad import get_speech_timestamps

import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
from preprocessing import Wav2Mel
from scipy.spatial.distance import pdist, squareform

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
dvector = torch.jit.load("custom_pipeline/dvector-step250000.pt").eval()

# vad model
vad_model = torch.jit.load("custom_pipeline/silero_vad.jit")


print("loading audio")
# read audio from file
wav_tensor, native_sample_rate = torchaudio.load("custom_pipeline/gwtwd.wav")
# resample to 16 kHz, apply effects and compute mel spectrogram
wav_tensor, mel_tensor = wav2mel(wav_tensor, native_sample_rate)

print("analyzing audio")
# Use pretrained vad model to find intervals containing speech
#speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate, return_seconds=True)
#pprint(speech_timestamps)


def get_frames(mel_tensor, block_size):
    return mel_tensor.unfold(0, block_size, block_size).mT


def get_frame_embeddings(mel_frames):
    embeddings = torch.empty(mel_frames.shape[0], 256)
    for frame_idx in range(mel_frames.shape[0]):
        embeddings[frame_idx, :] = dvector.embed_utterance(mel_frames[frame_idx])
    return embeddings.detach().numpy()


block_size = 50 # block size is the number of MFCC frames to stack togeter for each embeding frame
mel_frames = get_frames(mel_tensor, block_size) 
embed = get_frame_embeddings(mel_frames)

print(mel_frames.shape)
print(embed.shape)
print(embed)

# Unfortunately, it is not easy to verify that the above procedure is correct.
# But the least you can do is having a look at the dimensions and ensuring they look ok.

# The next step is to cluster the embedings 
distance = pdist(embed, metric='euclidean')
A = 1 / (1 + distance)
A = squareform(A)
print('A', A.shape)
print('Distance', distance.shape)
A = A - np.diag(np.diag(A)) # set diagonal to zero
plt.imshow(A)
plt.colorbar() 
plt.show()
def normalized_laplacian_matrix(A):
    # A is an adjacency matrix

    D = np.sum(A, axis=1)
    inv_sqrt_D = np.power(D, -0.5)
    L = inv_sqrt_D.reshape(-1, 1) * A * inv_sqrt_D.reshape(1, -1)

    return L

L = normalized_laplacian_matrix(A)
# Add colorbar
plt.imshow(L)
plt.colorbar()
plt.show()

w, v = np.linalg.eig(L)

eigenvalues = np.real(w)
#eigenvalues = eigenvalues[eigenvalues>0.05]
#eigenvectors = np.real(v[:, 0:len(eigenvalues)])
#print(np.diff(eigenvalues))
num_vectors = np.argmin(np.diff(eigenvalues)) + 1
num_vectors = 3
eigenvectors = np.real(v[:, 0:num_vectors])
plt.scatter(range(len(eigenvalues)), eigenvalues)
plt.show()

norm_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

plt.scatter(norm_eigenvectors[:, 0], norm_eigenvectors[:, 1])
plt.show()
plt.scatter(norm_eigenvectors[:, 1], norm_eigenvectors[:, 2])
plt.show()

predictions = KMeans(n_clusters=norm_eigenvectors.shape[1], n_init=5).fit_predict(norm_eigenvectors)
plt.plot(predictions)
plt.show()

# TODO:
# RTTM format needs to be fixed!
# If there is no voice, our model needs to detect that. Incorporate the VAD time stamps into the model.
# We need to find out whether we want to incorporate the time stamps before or after the clustering.