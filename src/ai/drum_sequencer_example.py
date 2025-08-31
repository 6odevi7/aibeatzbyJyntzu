# Example: Beat sequencing using pydub and drumkit samples
from pydub import AudioSegment
import os, random

def sequence_beat(drumkit_path, pattern=["kick","snare","hihat"], bpm=120, bars=4):
    # Find samples
    samples = {name: None for name in pattern}
    for file in os.listdir(drumkit_path):
        fname = file.lower()
        for name in pattern:
            if name in fname and file.endswith('.wav'):
                samples[name] = AudioSegment.from_wav(os.path.join(drumkit_path, file))
    beat = AudioSegment.silent(duration=bars*4*60000//bpm)
    # Simple pattern: place samples at regular intervals
    for i in range(bars*4):
        pos = i * 60000 // bpm
        if samples["kick"]: beat = beat.overlay(samples["kick"], position=pos)
        if i%2==0 and samples["snare"]: beat = beat.overlay(samples["snare"], position=pos)
        if samples["hihat"]: beat = beat.overlay(samples["hihat"], position=pos)
    out_path = os.path.join(drumkit_path, "generated_beat.wav")
    beat.export(out_path, format="wav")
    return out_path

# Usage:
# sequence_beat("../../drumkits/unpacked/SomeDrumkitFolder")
