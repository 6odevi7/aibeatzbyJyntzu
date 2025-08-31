import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import random
import json
from pathlib import Path

class SampleChopper:
    def __init__(self):
        self.audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a'}
        self.chop_cache = {}
        
    def load_audio(self, file_path):
        """Load audio file and convert to numpy array"""
        try:
            audio = AudioSegment.from_file(file_path)
            # Convert to mono for analysis
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Get raw audio data
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            
            return samples, sample_rate, audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, None
    
    def detect_transients(self, samples, sample_rate, sensitivity=0.3):
        """Detect transients/beats in audio for intelligent chopping"""
        # Use librosa for onset detection
        onset_frames = librosa.onset.onset_detect(
            y=samples, 
            sr=sample_rate, 
            units='time',
            hop_length=512,
            backtrack=True,
            pre_max=20,
            post_max=20,
            pre_avg=100,
            post_avg=100,
            delta=sensitivity,
            wait=10
        )
        return onset_frames
    
    def detect_tempo_and_beats(self, samples, sample_rate):
        """Detect tempo and beat positions"""
        tempo, beats = librosa.beat.beat_track(
            y=samples, 
            sr=sample_rate, 
            units='time'
        )
        return tempo, beats
    
    def chop_by_transients(self, file_path, min_length=0.1, max_length=4.0):
        """Chop sample based on detected transients"""
        samples, sample_rate, audio = self.load_audio(file_path)
        if samples is None:
            return []
        
        # Detect transients
        onsets = self.detect_transients(samples, sample_rate)
        
        # Create chops between transients
        chops = []
        for i in range(len(onsets) - 1):
            start_time = onsets[i]
            end_time = onsets[i + 1]
            duration = end_time - start_time
            
            # Filter by length constraints
            if min_length <= duration <= max_length:
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                chop = audio[start_ms:end_ms]
                
                chops.append({
                    'audio': chop,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'index': i
                })
        
        return chops
    
    def chop_by_beats(self, file_path, bars=1):
        """Chop sample into beat-synchronized segments"""
        samples, sample_rate, audio = self.load_audio(file_path)
        if samples is None:
            return []
        
        # Detect tempo and beats
        tempo, beats = self.detect_tempo_and_beats(samples, sample_rate)
        
        # Calculate beats per bar (assuming 4/4 time)
        beats_per_bar = 4
        beat_length = beats_per_bar * bars
        
        chops = []
        for i in range(0, len(beats) - beat_length, beat_length):
            start_time = beats[i]
            end_time = beats[i + beat_length] if i + beat_length < len(beats) else len(samples) / sample_rate
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            chop = audio[start_ms:end_ms]
            
            chops.append({
                'audio': chop,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'bars': bars,
                'tempo': tempo,
                'index': i // beat_length
            })
        
        return chops
    
    def chop_equal_segments(self, file_path, num_segments=8):
        """Chop sample into equal segments"""
        samples, sample_rate, audio = self.load_audio(file_path)
        if samples is None:
            return []
        
        duration_ms = len(audio)
        segment_length = duration_ms // num_segments
        
        chops = []
        for i in range(num_segments):
            start_ms = i * segment_length
            end_ms = (i + 1) * segment_length
            
            if end_ms > duration_ms:
                end_ms = duration_ms
            
            chop = audio[start_ms:end_ms]
            
            chops.append({
                'audio': chop,
                'start_time': start_ms / 1000,
                'end_time': end_ms / 1000,
                'duration': (end_ms - start_ms) / 1000,
                'index': i
            })
        
        return chops
    
    def apply_effects(self, audio_segment, effects=None):
        """Apply effects to audio segment"""
        if effects is None:
            effects = {}
        
        processed = audio_segment
        
        # Normalize
        if effects.get('normalize', False):
            processed = normalize(processed)
        
        # Compress
        if effects.get('compress', False):
            processed = compress_dynamic_range(processed)
        
        # Reverse
        if effects.get('reverse', False):
            processed = processed.reverse()
        
        # Pitch shift (simple speed change)
        if 'pitch_shift' in effects:
            shift = effects['pitch_shift']
            if shift != 0:
                new_sample_rate = int(processed.frame_rate * (2 ** (shift / 12)))
                processed = processed._spawn(processed.raw_data, overrides={"frame_rate": new_sample_rate})
                processed = processed.set_frame_rate(44100)
        
        # Volume adjustment
        if 'volume' in effects:
            volume_db = effects['volume']
            processed = processed + volume_db
        
        # Fade in/out
        if 'fade_in' in effects:
            processed = processed.fade_in(int(effects['fade_in'] * 1000))
        
        if 'fade_out' in effects:
            processed = processed.fade_out(int(effects['fade_out'] * 1000))
        
        return processed
    
    def smart_chop(self, file_path, method='auto', **kwargs):
        """Intelligently chop sample based on content analysis"""
        samples, sample_rate, audio = self.load_audio(file_path)
        if samples is None:
            return []
        
        # Analyze audio characteristics
        tempo, beats = self.detect_tempo_and_beats(samples, sample_rate)
        onsets = self.detect_transients(samples, sample_rate)
        
        # Determine best chopping method
        if method == 'auto':
            # If strong rhythmic content, use beat-based chopping
            if len(beats) > 8 and tempo > 60:
                method = 'beats'
            # If many transients, use transient-based chopping
            elif len(onsets) > 4:
                method = 'transients'
            # Otherwise, use equal segments
            else:
                method = 'equal'
        
        # Apply chosen method
        if method == 'beats':
            bars = kwargs.get('bars', 1)
            return self.chop_by_beats(file_path, bars)
        elif method == 'transients':
            min_length = kwargs.get('min_length', 0.1)
            max_length = kwargs.get('max_length', 4.0)
            return self.chop_by_transients(file_path, min_length, max_length)
        elif method == 'equal':
            num_segments = kwargs.get('num_segments', 8)
            return self.chop_equal_segments(file_path, num_segments)
        
        return []
    
    def save_chops(self, chops, output_dir, base_name, effects=None):
        """Save chopped segments to files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for i, chop in enumerate(chops):
            # Apply effects if specified
            audio = chop['audio']
            if effects:
                audio = self.apply_effects(audio, effects)
            
            # Generate filename
            filename = f"{base_name}_chop_{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Export
            audio.export(filepath, format="wav")
            
            # Store metadata
            chop_info = {
                'file': filepath,
                'start_time': chop['start_time'],
                'end_time': chop['end_time'],
                'duration': chop['duration'],
                'index': chop['index']
            }
            
            if 'tempo' in chop:
                chop_info['tempo'] = chop['tempo']
            if 'bars' in chop:
                chop_info['bars'] = chop['bars']
            
            saved_files.append(chop_info)
        
        return saved_files
    
    def create_chop_variations(self, chop_audio, num_variations=3):
        """Create variations of a chop with different effects"""
        variations = []
        
        effect_presets = [
            {'normalize': True, 'volume': 2},
            {'reverse': True, 'fade_in': 0.1},
            {'pitch_shift': random.uniform(-2, 2), 'compress': True},
            {'volume': random.uniform(-6, 6), 'fade_out': 0.2},
            {'pitch_shift': random.choice([-12, -7, -5, 5, 7, 12]), 'normalize': True}
        ]
        
        # Original
        variations.append(chop_audio)
        
        # Create variations
        for i in range(min(num_variations, len(effect_presets))):
            effects = effect_presets[i]
            variation = self.apply_effects(chop_audio, effects)
            variations.append(variation)
        
        return variations
    
    def batch_chop_folder(self, input_folder, output_folder, method='auto', create_variations=False):
        """Batch process all audio files in a folder"""
        results = {}
        
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if Path(file).suffix.lower() in self.audio_exts:
                    file_path = os.path.join(root, file)
                    base_name = Path(file).stem
                    
                    print(f"Processing: {file}")
                    
                    # Chop the file
                    chops = self.smart_chop(file_path, method=method)
                    
                    if chops:
                        # Create output directory for this file
                        file_output_dir = os.path.join(output_folder, base_name)
                        
                        # Save original chops
                        saved_chops = self.save_chops(chops, file_output_dir, base_name)
                        
                        # Create variations if requested
                        if create_variations:
                            var_dir = os.path.join(file_output_dir, 'variations')
                            for i, chop in enumerate(chops):
                                variations = self.create_chop_variations(chop['audio'])
                                for j, var in enumerate(variations[1:], 1):  # Skip original
                                    var_name = f"{base_name}_chop_{i:02d}_var_{j}"
                                    var_path = os.path.join(var_dir, f"{var_name}.wav")
                                    os.makedirs(var_dir, exist_ok=True)
                                    var.export(var_path, format="wav")
                        
                        results[file] = {
                            'chops': saved_chops,
                            'total_chops': len(chops),
                            'output_dir': file_output_dir
                        }
        
        # Save processing report
        report_path = os.path.join(output_folder, 'chop_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    chopper = SampleChopper()
    
    # Test with a sample file
    test_file = "path/to/your/sample.wav"
    
    if os.path.exists(test_file):
        # Smart chop with auto detection
        chops = chopper.smart_chop(test_file, method='auto')
        
        # Save chops
        output_dir = "chopped_samples"
        base_name = Path(test_file).stem
        saved_files = chopper.save_chops(chops, output_dir, base_name)
        
        print(f"Created {len(saved_files)} chops from {test_file}")
        for chop in saved_files:
            print(f"  - {chop['file']} ({chop['duration']:.2f}s)")