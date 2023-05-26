
# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch import nn


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
    
    return ''.join([index2char[label] for label in labels])


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
    spectrograms = [transform(batch[0]) for batch in data]
    labels = [strToInt(batch[2]) for batch in data]
    input_lengths = [spectrogram.shape[2]//2 for spectrogram in spectrograms]
    label_lengths = [len(label) for label in labels]
    return spectrograms, labels, input_lengths, label_lengths

import torch
example = torch.load('lab4/lab4_example.pt')
print("Printing example:")
print(example["spectrograms"][0])
Y = dataProcessing(example["data"], train_audio_transform)
print("Printing Y:")
print(Y[0][0])

print("Printing example:")
print(example["labels"][0])
Y = dataProcessing(example["data"], train_audio_transform)
print("Printing Y:")
print(Y[1][0])
# Our length is int, theirs is float

print("Printing example:")
print(example["input_lengths"])
Y = dataProcessing(example["data"], train_audio_transform)
print("Printing Y:")
print(Y[2])

print("Printing example:")
print(example["label_lengths"])
Y = dataProcessing(example["data"], train_audio_transform)
print("Printing Y:")
print(Y[3])
    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''

def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
