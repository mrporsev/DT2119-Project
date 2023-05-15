# pylint: disable=missing-function-docstring
'''
This file contains the finished speaker diarization pipeline from the PyAnnote. 
It passes all data into the pipeline and outputs the data into a .rttm format.
'''

import torch
import distutils.version
import os
from pyannote.audio import Pipeline

def check_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"

    )
    print(f"Using {device} device")

def print_output_in_console(diarization_itertracks):
    # Iterate over the speaker segments and print the speaker label and corresponding time intervals
    for segment, _, speaker in diarization_itertracks:
        start_time = segment.start
        end_time = segment.end
        duration = end_time - start_time
        print(f"Speaker {speaker}: {start_time:.2f}s to {end_time:.2f}s (Duration: {duration:.2f}s)")

def count_files_in_directory(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)


def run_all_audio_files(directory, write_to_path, pl):

    number_of_files = count_files_in_directory(directory)
    print(f'Number of .wav files to send to pipeline: {number_of_files}')
    curr_num = 1
    for filename in os.listdir(directory):
        try:
            file_path = os.path.join(directory, filename)
            diarization = pl(file_path)
            curr_num += 1
            if not os.path.exists(write_to_path):
                os.makedirs(write_to_path)
            # Comment out the following to lines if you do not
            # want the output to be printed
            diarization_itertracks = diarization.itertracks(yield_label=True)
            print_output_in_console(diarization_itertracks)
            new_filename = filename.replace('.wav', '') + '_p.rttm'
            output_file = os.path.join(write_to_path, new_filename)
            # dump the diarization output to disk using RTTM format
            with open(output_file, "w", encoding='utf-8') as rttm:
                diarization.write_rttm(rttm)
            percentage = float(curr_num / number_of_files) * 100
            print(f'Finished processing file {curr_num} of {number_of_files}')
            print(f'Percentage done: {percentage:.2f}%')
                  
        except IOError as file_error:
            print(f'Could not process file: {filename}. Error: {file_error}')


pipeline = None
try:
    print('Loading pipeline from pyannote...')
    # If token do not work, create a new one!
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
except ImportError:
    print('Could not load the pipeline! Check if pyannote.audio and torch is installed.')

check_device()

# run the pipeline on all audio files in the directory
if pipeline is not None:
    run_all_audio_files("./Dataset/audio", "./Dataset/output", pl = pipeline)
else:
    print('Pipeline was not loaded!')

# Example usage! Uncomment following line to apply the pipeline to one audio file
# diarization = pipeline("./Dataset/audio/ampme.wav")