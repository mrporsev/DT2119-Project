
# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch import nn
from torchaudio.functional import edit_distance
from pyctcdecode import build_ctcdecoder


train_audio_transform = nn.Sequential(MelSpectrogram(n_mels=80), FrequencyMasking(15), TimeMasking(35))
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = nn.Sequential(MelSpectrogram(n_mels=80))

# Functions to be implemented ----------------------------------

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    vocab = "' abcdefghijklmnopqrstuvwxyz"
    index2char = [char for char in vocab]
    #print(index2char)
    
    return ''.join([index2char[int(label)] for label in labels])


def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    text = text.lower()
    vocab = "' abcdefghijklmnopqrstuvwxyz"
    index2char = [char for char in vocab]
    #print(index2char)
    char2index = {char: index for index, char in enumerate(index2char)}
    #print(char2index)

    return [char2index[char] for char in text]


x = strToInt("hello world'")
print(x)
x = np.array(x, dtype=np.float32)
y = intToStr(x)
print(y)
print()
def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    spectrograms_list = [transform(batch[0]).squeeze(0).transpose(0, 1) for batch in data]
    input_lengths = [spectrogram.shape[0]//2 for spectrogram in spectrograms_list]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms_list, batch_first=True).unsqueeze(1).transpose(2, 3)
    # spectrograms = torch.empty(len(data), 1, 80, 0)
    # for batch_idx, batch in enumerate(data):
    #     spectrograms[batch_idx][0] = transform(batch[0])

    labels_list = [torch.Tensor(strToInt(batch[2])) for batch in data]
    label_lengths = [len(label) for label in labels_list]
    labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths

import torch
example = torch.load('lab4/lab4_example.pt')
print(example["spectrograms"].shape)
Y = dataProcessing(example["data"], test_audio_transform)
print("Printing example:")
print(example["spectrograms"][1])
print("Printing Y:")
print(Y[0][1])
for spec_idx, spec in enumerate(example["spectrograms"]):
    if np.nanmax(np.abs(spec - Y[0][spec_idx])) > 1e-7:
        print("Index: ", spec_idx)
        print("Example: ", spec)
        print("Y: ", Y[0][spec_idx])
        print("Difference: ", spec[np.abs(spec-Y[0][spec_idx]) > 1e-7])
        print("Our difference: ", Y[0][spec_idx][np.abs(spec-Y[0][spec_idx]) > 1e-7])

#print("Y minus example:", (Y[0][1]- example["spectrograms"][1]))

print("Printing example:")
print(example["labels"][0])
print("Printing Y:")
print(Y[1][0])
# Our length is int, theirs is float

print("Printing example:")
print(example["input_lengths"])
print("Printing Y:")
print(Y[2])

print("Printing example:")
print(example["label_lengths"])
print("Printing Y:")
print(Y[3])

print("Comparing spectrograms absolute error: ", np.nanmax(np.abs(Y[0]- example["spectrograms"])))
print("Comparing spectrograms relative error: ", np.nanmax(np.abs(Y[0]- example["spectrograms"]) / (1e-10 + np.abs(Y[0]) + np.abs(example["spectrograms"]))))
print("Comparing labels: ", np.nanmax(np.abs(torch.Tensor(Y[1])- example["labels"])))
print("Comparing input_lengths: ", np.nanmax(np.abs(np.array(Y[2])- np.array(example["input_lengths"]))))
print("Comparing label_lengths: ", np.nanmax(np.abs(np.array(Y[3])- np.array(example["label_lengths"]))))

def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''

    decoded_strings = []
    for batch_idx in range(output.shape[0]):
        character_indeces = []
        last_label = blank_label
        for time_idx in range(output.shape[1]):
            label = output[batch_idx][time_idx].argmax()
            if label != last_label and label != blank_label:
                character_indeces.append(label)
            last_label = label
        decoded_strings.append(intToStr(character_indeces))

    return decoded_strings


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''

    matrix = np.empty((len(ref)+1, len(hyp)+1), dtype=int)
    matrix[0, :] = np.arange(len(hyp) + 1) # first row
    matrix[:, 0] = np.arange(len(ref) + 1) # first column
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i - 1] == hyp[j - 1]:
                matrix[i, j] = matrix[i-1, j-1]
            else:
                matrix[i, j] = min(matrix[i-1, j-1], matrix[i-1, j], matrix[i, j-1]) + 1
    return matrix[len(ref), len(hyp)]

a = "hello world"
b = "hello world"
print(levenshteinDistance(a, b))
print(edit_distance(a, b))


def languageDecoder(output, blank_label=28):

    labels = [c for c in "' abcdefghijklmnopqrstuvwxyz"]
    model = 'lab4/wiki-interpolate.3gram.arpa'
    print("model path: ", model)
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path="wiki-interpolate.3gram.arpa",
        alpha=0.5,
        beta=1,
    )

    decoded_strings = []
    for batch_idx in range(output.shape[0]):
        text = decoder.decode(output[batch_idx].cpu().detach().numpy())
        decoded_strings.append(text)

    return decoded_strings

# TODO:
# 1. Modify training code to save progress to list
# 2. Train the model using a google colab GPU
# 3. Save results from greedy decoder and language decoder
# 3. Grid search for alpha and beta