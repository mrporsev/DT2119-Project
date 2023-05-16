'''
This file should contains the custom speaker diarization pipeline with modules from pyAnnote. 
It should pass all data into the pipeline and outputs the data into a .rttm format.
'''
#We should create custom pipeline with custom clustering module, also fix a custom speaker embedding model for d-vectors. 
# Use the speaker embedding for D-vectors!!!, Github som heter D-vector with pretrained model.

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


from pyannote.audio.tasks import VoiceActivityDetection
from pyannote.audio import Model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

class CustomDiarization(Pipeline):
    def __init__(self):
        super().__init__()

        # load pre-trained VAD model
        self.vad = VoiceActivityDetection.from_pretrained("pyannote/vad",
                                                          use_auth_token="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")

        # load pre-trained speaker embedding model
        self.speaker_embedding = Model.from_pretrained("pyannote/embedding",
                                                       use_auth_token="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")
        
        # Binarize VAD scores with a 0.5 threshold.
        self.binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, min_duration_off=0.1, min_duration_on=0.1)

        # Diariozation error rate
        self.diarization_error_rate = {"collar": 0.0, "skip_overlap": False}

    def apply_vad(self, audio):
                # Apply VAD
        vad_scores = self.vad(audio)
        speech = self.binarize.apply(vad_scores, dimension=1)

        # Apply speech embedding
        embeddings = self.speaker_embedding(audio)

        # Transform embeddings to match VAD segments
        vad_embeddings = embeddings.crop(speech)

        # Perform your custom spectral clustering on the embeddings (OUR CUSTOM SPECTRAL CLUSTERING ALGORITHM)
        clusters = my_spectral_clustering(vad_embeddings.data)

        # Convert the clusters back into pyannote.core.Annotation format and return
        return SlidingWindowFeature(clusters, vad_embeddings.sliding_window)
    
    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
    
        
# ------------------ MAIN ------------------
pipeline = None
try:
    print('Loading custom pipeline ...')
    # If token do not work, create a new one!
    pipeline = CustomDiarization()
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