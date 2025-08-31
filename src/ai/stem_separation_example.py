# Example: Safe stem separation using Spleeter
import os
from spleeter.separator import Separator

def separate_stems(input_path, output_dir):
    separator = Separator('spleeter:2stems')  # vocals + accompaniment
    separator.separate_to_file(input_path, output_dir)
    return output_dir

# Usage:
# separate_stems('path/to/track.wav', 'path/to/output_folder')
