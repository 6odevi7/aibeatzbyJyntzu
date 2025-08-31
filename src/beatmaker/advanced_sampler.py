import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range, low_pass_filter, high_pass_filter
import librosa
import random
from pathlib import Path
from sample_chopper import SampleChopper
from sample_manager import SampleManager

class AdvancedSampler:
    def __init__(self, sample_manager=None):
        self.chopper = SampleChopper()
        self.manager = sample_manager or SampleManager()
        self.audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a'}
        
    def time_stretch(self, audio_segment, stretch_factor):
        """Time stretch audio without changing pitch using phase vocoder"""
        try:
            # Convert to numpy
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels > 1:
                samples = samples.reshape((-1, audio_segment.channels))
                samples = samples.mean(axis=1)
            
            # Apply time stretching
            stretched = librosa.effects.time_stretch(samples, rate=stretch_factor)
            
            # Convert back to AudioSegment
            stretched_int = (stretched * 32767).astype(np.int16)
            stretched_audio = AudioSegment(
                stretched_int.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=2,
                channels=1
            )
            
            return stretched_audio
        except Exception as e:
            print(f"Time stretch error: {e}")
            return audio_segment
    
    def pitch_shift(self, audio_segment, semitones):
        """Pitch shift audio using librosa"""
        try:
            # Convert to numpy
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels > 1:
                samples = samples.reshape((-1, audio_segment.channels))
                samples = samples.mean(axis=1)
            
            # Apply pitch shifting
            shifted = librosa.effects.pitch_shift(
                samples, 
                sr=audio_segment.frame_rate, 
                n_steps=semitones
            )
            
            # Convert back to AudioSegment
            shifted_int = (shifted * 32767).astype(np.int16)
            shifted_audio = AudioSegment(
                shifted_int.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=2,
                channels=1
            )
            
            return shifted_audio
        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio_segment
    
    def granular_synthesis(self, audio_segment, grain_size_ms=50, overlap=0.5, randomize=True):
        """Apply granular synthesis effects"""
        try:
            grain_size = int(grain_size_ms)
            overlap_size = int(grain_size * overlap)
            
            # Create grains
            grains = []
            pos = 0
            
            while pos < len(audio_segment) - grain_size:
                grain = audio_segment[pos:pos + grain_size]
                
                if randomize:
                    # Random pitch shift
                    pitch_shift_factor = random.uniform(0.8, 1.2)
                    grain = grain._spawn(
                        grain.raw_data,
                        overrides={"frame_rate": int(grain.frame_rate * pitch_shift_factor)}
                    ).set_frame_rate(audio_segment.frame_rate)
                    
                    # Random volume
                    volume_change = random.uniform(-6, 6)
                    grain = grain + volume_change
                
                grains.append(grain)
                pos += grain_size - overlap_size
            
            # Reconstruct audio
            if not grains:
                return audio_segment
            
            result = grains[0]
            for grain in grains[1:]:
                result = result.overlay(grain, position=len(result) - overlap_size)
            
            return result
        except Exception as e:
            print(f"Granular synthesis error: {e}")
            return audio_segment
    
    def apply_swing(self, audio_segment, swing_amount=0.1):
        """Apply swing timing to audio"""
        try:
            # Detect beats
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels > 1:
                samples = samples.reshape((-1, audio_segment.channels))
                samples = samples.mean(axis=1)
            
            tempo, beats = librosa.beat.beat_track(
                y=samples, 
                sr=audio_segment.frame_rate, 
                units='time'
            )
            
            if len(beats) < 4:
                return audio_segment
            
            # Apply swing to off-beats
            swung_audio = AudioSegment.empty()
            
            for i in range(len(beats) - 1):
                beat_start = int(beats[i] * 1000)
                beat_end = int(beats[i + 1] * 1000)
                beat_segment = audio_segment[beat_start:beat_end]
                
                # Apply swing to every other beat
                if i % 2 == 1:
                    # Delay the beat slightly
                    delay_ms = int(swing_amount * (beat_end - beat_start))
                    silence = AudioSegment.silent(duration=delay_ms)
                    beat_segment = silence + beat_segment[:-delay_ms]
                
                swung_audio += beat_segment
            
            return swung_audio
        except Exception as e:
            print(f"Swing error: {e}")
            return audio_segment
    
    def create_stutter_effect(self, audio_segment, stutter_length_ms=125, repeats=4):
        """Create stutter/glitch effects"""
        try:
            stutter_length = int(stutter_length_ms)
            
            # Find a good stutter point (usually at transients)
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels > 1:
                samples = samples.reshape((-1, audio_segment.channels))
                samples = samples.mean(axis=1)
            
            # Detect onsets
            onsets = librosa.onset.onset_detect(
                y=samples,
                sr=audio_segment.frame_rate,
                units='time'
            )
            
            if len(onsets) == 0:
                stutter_start = len(audio_segment) // 2
            else:
                # Pick a random onset
                stutter_start = int(random.choice(onsets) * 1000)
            
            # Create stutter
            stutter_segment = audio_segment[stutter_start:stutter_start + stutter_length]
            
            # Build stuttered version
            result = audio_segment[:stutter_start]
            
            for i in range(repeats):
                # Vary the stutter slightly
                variation = stutter_segment
                if i > 0:
                    # Apply slight pitch variation
                    pitch_factor = random.uniform(0.95, 1.05)
                    variation = variation._spawn(
                        variation.raw_data,
                        overrides={"frame_rate": int(variation.frame_rate * pitch_factor)}
                    ).set_frame_rate(audio_segment.frame_rate)
                
                result += variation
            
            # Add the rest of the original
            result += audio_segment[stutter_start + stutter_length:]
            
            return result
        except Exception as e:
            print(f"Stutter effect error: {e}")
            return audio_segment
    
    def create_reverse_reverb(self, audio_segment, reverb_length_ms=500):
        """Create reverse reverb effect"""
        try:
            # Reverse the audio
            reversed_audio = audio_segment.reverse()
            
            # Apply reverb (simple delay-based reverb)
            reverb = AudioSegment.empty()
            delays = [50, 100, 150, 200, 300, 400]
            
            for delay in delays:
                delayed = AudioSegment.silent(duration=delay) + reversed_audio
                delayed = delayed - (6 + delay // 50)  # Reduce volume with each delay
                if len(reverb) == 0:
                    reverb = delayed
                else:
                    reverb = reverb.overlay(delayed)
            
            # Reverse back and combine
            reverb = reverb.reverse()
            
            # Trim to reverb length
            reverb_length = int(reverb_length_ms)
            reverb = reverb[:reverb_length]
            
            # Combine with original
            result = reverb + audio_segment
            
            return result
        except Exception as e:
            print(f"Reverse reverb error: {e}")
            return audio_segment
    
    def slice_and_dice(self, audio_segment, num_slices=8, randomize_order=True):
        """Slice audio and rearrange pieces"""
        try:
            slice_length = len(audio_segment) // num_slices
            slices = []
            
            for i in range(num_slices):
                start = i * slice_length
                end = start + slice_length
                if i == num_slices - 1:  # Last slice gets remainder
                    end = len(audio_segment)
                
                slice_audio = audio_segment[start:end]
                slices.append(slice_audio)
            
            if randomize_order:
                random.shuffle(slices)
            
            # Reconstruct
            result = AudioSegment.empty()
            for slice_audio in slices:
                result += slice_audio
            
            return result
        except Exception as e:
            print(f"Slice and dice error: {e}")
            return audio_segment
    
    def create_filter_sweep(self, audio_segment, filter_type='low', start_freq=200, end_freq=8000):
        """Create filter sweep effect"""
        try:
            duration_ms = len(audio_segment)
            result = AudioSegment.empty()
            
            # Process in small chunks
            chunk_size = 100  # ms
            num_chunks = duration_ms // chunk_size
            
            for i in range(num_chunks):
                start_ms = i * chunk_size
                end_ms = min(start_ms + chunk_size, duration_ms)
                chunk = audio_segment[start_ms:end_ms]
                
                # Calculate frequency for this chunk
                progress = i / num_chunks
                current_freq = start_freq + (end_freq - start_freq) * progress
                
                # Apply filter
                if filter_type == 'low':
                    filtered_chunk = low_pass_filter(chunk, current_freq)
                else:  # high pass
                    filtered_chunk = high_pass_filter(chunk, current_freq)
                
                result += filtered_chunk
            
            # Add any remaining audio
            if len(result) < duration_ms:
                result += audio_segment[len(result):]
            
            return result
        except Exception as e:
            print(f"Filter sweep error: {e}")
            return audio_segment
    
    def intelligent_chop_and_flip(self, file_path, style='boom_bap'):
        """Intelligently chop and flip samples based on style"""
        # Load and analyze the sample
        samples, sample_rate, audio = self.chopper.load_audio(file_path)
        if samples is None:
            return []
        
        # Detect musical elements
        tempo, beats = librosa.beat.beat_track(y=samples, sr=sample_rate, units='time')
        onsets = self.chopper.detect_transients(samples, sample_rate)
        
        variations = []
        base_name = Path(file_path).stem
        
        if style == 'boom_bap':
            # Classic boom bap chopping
            # 1. Chop by beats (1-2 bar loops)
            beat_chops = self.chopper.chop_by_beats(file_path, bars=1)
            for i, chop in enumerate(beat_chops[:4]):  # Take first 4
                # Apply classic effects
                processed = chop['audio']
                processed = normalize(processed)
                processed = compress_dynamic_range(processed)
                
                # Add some vinyl crackle simulation (high-pass filter + noise)
                processed = high_pass_filter(processed, 80)
                
                variations.append({
                    'audio': processed,
                    'name': f"{base_name}_boom_bap_{i+1}",
                    'style': 'boom_bap',
                    'type': 'beat_chop'
                })
        
        elif style == 'trap':
            # Modern trap chopping
            # 1. Shorter chops with pitch variations
            transient_chops = self.chopper.chop_by_transients(file_path, min_length=0.2, max_length=1.0)
            for i, chop in enumerate(transient_chops[:6]):
                processed = chop['audio']
                
                # Apply trap-style effects
                processed = normalize(processed)
                
                # Pitch variations
                if i % 2 == 0:
                    processed = self.pitch_shift(processed, random.choice([-12, -7, 7, 12]))
                
                # Add stutter occasionally
                if random.random() < 0.3:
                    processed = self.create_stutter_effect(processed)
                
                variations.append({
                    'audio': processed,
                    'name': f"{base_name}_trap_{i+1}",
                    'style': 'trap',
                    'type': 'transient_chop'
                })
        
        elif style == 'drill':
            # UK/Chicago drill style
            # Aggressive chopping with slides and pitch bends
            equal_chops = self.chopper.chop_equal_segments(file_path, num_segments=8)
            for i, chop in enumerate(equal_chops):
                processed = chop['audio']
                
                # Drill-style processing
                processed = normalize(processed)
                processed = compress_dynamic_range(processed)
                
                # Pitch slides (simulate 808 slides)
                if i % 3 == 0:
                    # Create pitch bend effect
                    processed = self.pitch_shift(processed, random.uniform(-5, 5))
                
                # Reverse some chops
                if random.random() < 0.25:
                    processed = processed.reverse()
                
                variations.append({
                    'audio': processed,
                    'name': f"{base_name}_drill_{i+1}",
                    'style': 'drill',
                    'type': 'equal_chop'
                })
        
        elif style == 'experimental':
            # Experimental/glitch chopping
            # Heavy processing and manipulation
            chops = self.chopper.smart_chop(file_path, method='transients')
            for i, chop in enumerate(chops[:8]):
                processed = chop['audio']
                
                # Apply experimental effects
                effects = [
                    lambda x: self.granular_synthesis(x, randomize=True),
                    lambda x: self.slice_and_dice(x, randomize_order=True),
                    lambda x: self.create_filter_sweep(x),
                    lambda x: self.time_stretch(x, random.uniform(0.5, 2.0)),
                    lambda x: self.create_reverse_reverb(x)
                ]
                
                # Apply 1-3 random effects
                num_effects = random.randint(1, 3)
                selected_effects = random.sample(effects, num_effects)
                
                for effect in selected_effects:
                    processed = effect(processed)
                
                variations.append({
                    'audio': processed,
                    'name': f"{base_name}_experimental_{i+1}",
                    'style': 'experimental',
                    'type': 'multi_effect'
                })
        
        return variations
    
    def create_sample_pack(self, source_files, output_dir, style='boom_bap', pack_name=None):
        """Create a complete sample pack from source files"""
        if pack_name is None:
            pack_name = f"sample_pack_{style}_{random.randint(1000, 9999)}"
        
        pack_dir = os.path.join(output_dir, pack_name)
        os.makedirs(pack_dir, exist_ok=True)
        
        all_variations = []
        
        for file_path in source_files:
            if Path(file_path).suffix.lower() in self.audio_exts:
                print(f"Processing: {Path(file_path).name}")
                
                # Create variations
                variations = self.intelligent_chop_and_flip(file_path, style=style)
                
                # Save variations
                for var in variations:
                    filename = f"{var['name']}.wav"
                    filepath = os.path.join(pack_dir, filename)
                    var['audio'].export(filepath, format="wav")
                    
                    # Add to database
                    sample_id = self.manager.add_sample(
                        filepath,
                        name=var['name'],
                        genre=style,
                        tags=[var['style'], var['type'], 'chopped', 'processed']
                    )
                    
                    all_variations.append({
                        'file': filepath,
                        'name': var['name'],
                        'style': var['style'],
                        'type': var['type'],
                        'sample_id': sample_id
                    })
        
        # Create pack info file
        pack_info = {
            'name': pack_name,
            'style': style,
            'created_date': str(datetime.now()),
            'total_samples': len(all_variations),
            'source_files': source_files,
            'samples': all_variations
        }
        
        info_path = os.path.join(pack_dir, 'pack_info.json')
        with open(info_path, 'w') as f:
            json.dump(pack_info, f, indent=2)
        
        print(f"Created sample pack: {pack_name}")
        print(f"Total samples: {len(all_variations)}")
        print(f"Output directory: {pack_dir}")
        
        return pack_info

# Example usage
if __name__ == "__main__":
    from datetime import datetime
    import json
    
    sampler = AdvancedSampler()
    
    # Example: Create a boom bap sample pack
    source_files = [
        # Add paths to your source samples here
        # "path/to/sample1.wav",
        # "path/to/sample2.wav",
    ]
    
    if source_files:
        pack_info = sampler.create_sample_pack(
            source_files,
            output_dir="sample_packs",
            style="boom_bap",
            pack_name="Classic_Boom_Bap_Pack"
        )
        
        print("Sample pack created successfully!")
    else:
        print("Add source files to test the sample pack creation")