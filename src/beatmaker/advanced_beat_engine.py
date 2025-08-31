import random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy functions
    class np:
        @staticmethod
        def log10(x):
            import math
            return math.log10(x)

from pydub import AudioSegment
from pydub.generators import Sine, Square, Triangle, Sawtooth
import os
import json

class AdvancedBeatEngine:
    def __init__(self):
        self.complexity_levels = {
            'simple': {'layers': 3, 'variations': 2, 'effects': 1},
            'intermediate': {'layers': 5, 'variations': 4, 'effects': 2},
            'advanced': {'layers': 8, 'variations': 6, 'effects': 4},
            'expert': {'layers': 12, 'variations': 8, 'effects': 6}
        }
        
        self.genre_templates = {
            'trap': {
                'kick_pattern': [1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0],
                'snare_pattern': [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1],
                'hihat_pattern': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                '808_slides': True,
                'tempo_range': (130, 150),
                'key_signatures': ['F#m', 'C#m', 'Am', 'Dm']
            },
            'drill': {
                'kick_pattern': [1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0],
                'snare_pattern': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                'hihat_pattern': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                'slide_808s': True,
                'tempo_range': (140, 160),
                'key_signatures': ['Gm', 'Em', 'Bm', 'F#m']
            },
            'lo-fi': {
                'kick_pattern': [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
                'snare_pattern': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                'hihat_pattern': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                'vinyl_fx': True,
                'tempo_range': (70, 90),
                'key_signatures': ['C', 'Am', 'F', 'Dm']
            }
        }
    
    def generate_complex_beat(self, params, samples, user_preferences=None):
        """Generate complex beat with multiple layers and variations"""
        genre = params.get('genre', 'trap')
        complexity = params.get('complexity', 'intermediate')
        tempo = params.get('tempo', 120)
        length = params.get('length', 16)
        
        # Get complexity settings
        complexity_config = self.complexity_levels.get(complexity, self.complexity_levels['intermediate'])
        
        # Calculate timing
        beat_duration = int((60000 / tempo) * length)
        
        # Create base track
        final_beat = AudioSegment.silent(duration=beat_duration)
        
        # Generate multiple layers
        layers = []
        
        # Layer 1: Drum foundation
        drum_layer = self.create_drum_foundation(samples, genre, tempo, length, complexity_config)
        layers.append(('drums', drum_layer))
        
        # Layer 2: Bass/808 layer
        bass_layer = self.create_bass_layer(samples, genre, tempo, length, complexity_config)
        layers.append(('bass', bass_layer))
        
        # Layer 3: Melody layer
        melody_layer = self.create_melody_layer(samples, genre, tempo, length, complexity_config)
        layers.append(('melody', melody_layer))
        
        # Layer 4: Percussion fills
        if complexity_config['layers'] >= 5:
            perc_layer = self.create_percussion_layer(samples, genre, tempo, length)
            layers.append(('percussion', perc_layer))
        
        # Layer 5: Atmospheric layer
        if complexity_config['layers'] >= 8:
            atmo_layer = self.create_atmospheric_layer(samples, tempo, length)
            layers.append(('atmosphere', atmo_layer))
        
        # Mix all layers
        for layer_name, layer_audio in layers:
            if layer_audio and len(layer_audio) > 0:
                final_beat = final_beat.overlay(layer_audio)
        
        # Apply advanced effects
        final_beat = self.apply_advanced_effects(final_beat, complexity_config, genre)
        
        return final_beat
    
    def create_drum_foundation(self, samples, genre, tempo, length, complexity_config):
        """Create complex drum foundation with variations"""
        drum_samples = samples.get('kick', []) + samples.get('snare', []) + samples.get('hihat', [])
        if not drum_samples:
            return self.create_synthetic_drums(tempo, length, genre)
        
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        drum_track = AudioSegment.silent(duration=total_duration)
        
        # Get genre template
        template = self.genre_templates.get(genre, self.genre_templates['trap'])
        
        # Create variations for each bar
        variations = complexity_config['variations']
        
        bars = length // 4
        for bar in range(bars):
            # Apply pattern variations
            kick_pattern = self.vary_pattern(template['kick_pattern'], variations, bar)
            snare_pattern = self.vary_pattern(template['snare_pattern'], variations, bar)
            hihat_pattern = self.vary_pattern(template['hihat_pattern'], variations, bar)
            
            # Add drum hits with micro-timing
            for step in range(16):
                step_time = int((bar * 16 + step) * (beat_interval / 4))
                
                # Kick
                if kick_pattern[step] and samples.get('kick'):
                    kick_sample = random.choice(samples['kick'][:3])
                    kick_audio = AudioSegment.from_file(kick_sample['path'])[:800]
                    # Add micro-timing humanization
                    timing_offset = random.randint(-20, 20)
                    drum_track = drum_track.overlay(kick_audio, position=max(0, step_time + timing_offset))
                
                # Snare
                if snare_pattern[step] and samples.get('snare'):
                    snare_sample = random.choice(samples['snare'][:3])
                    snare_audio = AudioSegment.from_file(snare_sample['path'])[:600]
                    timing_offset = random.randint(-15, 15)
                    drum_track = drum_track.overlay(snare_audio, position=max(0, step_time + timing_offset))
                
                # Hi-hat
                if hihat_pattern[step] and samples.get('hihat'):
                    hihat_sample = random.choice(samples['hihat'][:3])
                    hihat_audio = AudioSegment.from_file(hihat_sample['path'])[:300]
                    # Velocity variation
                    velocity = random.uniform(0.6, 1.0)
                    if HAS_NUMPY:
                        hihat_audio = hihat_audio + (20 * np.log10(velocity))
                    else:
                        import math
                        hihat_audio = hihat_audio + (20 * math.log10(velocity))
                    timing_offset = random.randint(-10, 10)
                    drum_track = drum_track.overlay(hihat_audio, position=max(0, step_time + timing_offset))
        
        return drum_track
    
    def create_bass_layer(self, samples, genre, tempo, length, complexity_config):
        """Create complex bass layer with musical progressions"""
        bass_samples = samples.get('808', []) or samples.get('bass', [])
        if not bass_samples:
            return self.create_synthetic_bass(tempo, length, genre)
        
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        bass_track = AudioSegment.silent(duration=total_duration)
        
        # Musical chord progression
        progressions = {
            'trap': [0, -5, -3, -7],  # i-V-vi-IV
            'drill': [0, -2, -5, -3],  # i-VII-V-vi
            'lo-fi': [0, -3, -5, -7]   # i-vi-V-IV
        }
        
        chord_prog = progressions.get(genre, progressions['trap'])
        bars = length // 4
        
        for bar in range(bars):
            root_note = chord_prog[bar % len(chord_prog)]
            
            # Create bass pattern for this bar
            bass_pattern = self.generate_bass_pattern(complexity_config, bar)
            
            for step, (hit, pitch_offset) in enumerate(bass_pattern):
                if hit and bass_samples:
                    step_time = int((bar * 16 + step) * (beat_interval / 4))
                    
                    bass_sample = random.choice(bass_samples[:2])
                    bass_audio = AudioSegment.from_file(bass_sample['path'])[:1500]
                    
                    # Apply pitch shift for musical harmony
                    final_pitch = root_note + pitch_offset
                    bass_audio = self.pitch_shift_advanced(bass_audio, final_pitch)
                    
                    # Add bass processing
                    bass_audio = bass_audio + random.uniform(3, 8)  # Volume boost
                    
                    bass_track = bass_track.overlay(bass_audio, position=step_time)
        
        return bass_track
    
    def create_melody_layer(self, samples, genre, tempo, length, complexity_config):
        """Create complex melody layer with harmonic content"""
        melody_samples = samples.get('melody', [])
        if not melody_samples:
            return self.create_synthetic_melody(tempo, length, genre)
        
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        melody_track = AudioSegment.silent(duration=total_duration)
        
        # Create melodic phrases
        phrase_length = beat_interval * 4  # 4-beat phrases
        num_phrases = length // 4
        
        for phrase in range(num_phrases):
            if phrase % 2 == 0 or random.random() < 0.6:  # Not every phrase has melody
                phrase_start = phrase * phrase_length
                
                # Select melody samples for this phrase
                phrase_samples = random.sample(melody_samples, min(2, len(melody_samples)))
                
                for i, sample_info in enumerate(phrase_samples):
                    melody_audio = AudioSegment.from_file(sample_info['path'])[:3000]
                    
                    # Apply harmonic processing
                    melody_audio = melody_audio - random.uniform(8, 15)  # Reduce volume for background
                    
                    # Position within phrase
                    position = phrase_start + (i * (phrase_length // len(phrase_samples)))
                    
                    if position < total_duration:
                        melody_track = melody_track.overlay(melody_audio, position=position)
        
        return melody_track
    
    def create_percussion_layer(self, samples, genre, tempo, length):
        """Create additional percussion layer"""
        perc_samples = samples.get('perc', [])
        if not perc_samples:
            return AudioSegment.silent(duration=int((60000 / tempo) * length))
        
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        perc_track = AudioSegment.silent(duration=total_duration)
        
        # Add percussion hits at strategic points
        perc_positions = [
            int(total_duration * 0.25),
            int(total_duration * 0.5),
            int(total_duration * 0.75)
        ]
        
        for pos in perc_positions:
            if random.random() < 0.7:  # 70% chance
                perc_sample = random.choice(perc_samples)
                perc_audio = AudioSegment.from_file(perc_sample['path'])[:1000]
                perc_audio = perc_audio - random.uniform(5, 10)
                perc_track = perc_track.overlay(perc_audio, position=pos)
        
        return perc_track
    
    def create_atmospheric_layer(self, samples, tempo, length):
        """Create atmospheric background layer"""
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        
        # Create ambient pad
        pad_freq = random.choice([220, 330, 440])  # A3, E4, A4
        pad = Sine(pad_freq).to_audio_segment(duration=total_duration)
        pad = pad.apply_gain(-20)  # Very quiet background
        
        return pad
    
    def vary_pattern(self, base_pattern, variation_level, bar_number):
        """Create pattern variations based on complexity"""
        pattern = base_pattern.copy()
        
        # Add fills on certain bars
        if bar_number % 4 == 3:  # Fill on 4th bar
            for i in range(12, 16):  # Last quarter
                if random.random() < 0.4:
                    pattern[i] = 1
        
        # Add ghost notes based on variation level
        if variation_level >= 4:
            for i in range(len(pattern)):
                if pattern[i] == 0 and random.random() < 0.15:
                    pattern[i] = 0.5  # Ghost note
        
        return pattern
    
    def generate_bass_pattern(self, complexity_config, bar_number):
        """Generate bass pattern with pitch information"""
        patterns = [
            # Simple patterns
            [(1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 5), (0, 0),
             (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 3), (0, 0)],
            
            # Complex patterns
            [(1, 0), (0, 0), (1, 7), (0, 0), (1, 0), (0, 0), (1, 5), (0, 0),
             (1, 3), (0, 0), (1, 0), (0, 0), (1, 5), (0, 0), (1, 0), (0, 0)]
        ]
        
        if complexity_config['variations'] >= 4:
            return patterns[1]
        else:
            return patterns[0]
    
    def pitch_shift_advanced(self, audio, semitones):
        """Advanced pitch shifting simulation"""
        if semitones == 0:
            return audio
        
        # Simple rate-based pitch shift (approximation)
        rate_change = 2 ** (semitones / 12.0)
        
        if rate_change > 1:
            # Higher pitch - speed up
            return audio[::int(rate_change)] if len(audio) > 100 else audio
        else:
            # Lower pitch - slow down (approximate)
            return audio + audio[:int(len(audio) * (1 - rate_change))]
    
    def apply_advanced_effects(self, beat, complexity_config, genre):
        """Apply advanced audio effects"""
        effects_count = complexity_config['effects']
        
        # Compression (always applied)
        beat = beat.compress_dynamic_range(threshold=-18.0, ratio=3.0)
        
        if effects_count >= 2:
            # EQ simulation
            if genre in ['trap', 'drill']:
                # Boost low end
                beat = beat + 2
            elif genre == 'lo-fi':
                # Warm, filtered sound
                beat = beat - 1
        
        if effects_count >= 4:
            # Stereo width
            beat = beat.pan(random.uniform(-0.1, 0.1))
        
        if effects_count >= 6:
            # Subtle saturation (gain staging)
            beat = beat.apply_gain(random.uniform(0.5, 1.5))
        
        # Final normalization
        beat = beat.normalize(headroom=0.1)
        
        return beat
    
    def create_synthetic_drums(self, tempo, length, genre):
        """Create synthetic drums when no samples available"""
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        drum_track = AudioSegment.silent(duration=total_duration)
        
        # Create synthetic drum sounds
        kick = Sine(60).to_audio_segment(duration=300).apply_gain(-8)
        snare = Square(200).to_audio_segment(duration=200).apply_gain(-10)
        hihat = Square(8000).to_audio_segment(duration=100).apply_gain(-15)
        
        # Apply basic patterns
        beats_count = total_duration // beat_interval
        
        for beat in range(beats_count):
            beat_pos = beat * beat_interval
            
            # Kick on 1 and 3
            if beat % 4 in [0, 2]:
                drum_track = drum_track.overlay(kick, position=beat_pos)
            
            # Snare on 2 and 4
            if beat % 4 in [1, 3]:
                drum_track = drum_track.overlay(snare, position=beat_pos)
            
            # Hi-hat every beat
            drum_track = drum_track.overlay(hihat, position=beat_pos)
        
        return drum_track
    
    def create_synthetic_bass(self, tempo, length, genre):
        """Create synthetic bass when no samples available"""
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        
        # Bass frequency based on genre
        freq_map = {'trap': 55, 'drill': 50, 'lo-fi': 65}
        base_freq = freq_map.get(genre, 55)
        
        bass = Sine(base_freq).to_audio_segment(duration=total_duration)
        bass = bass.apply_gain(-12)
        
        return bass
    
    def create_synthetic_melody(self, tempo, length, genre):
        """Create synthetic melody when no samples available"""
        beat_interval = int(60000 / tempo)
        total_duration = beat_interval * length
        
        # Melody frequency based on genre
        freq_map = {'trap': 220, 'drill': 200, 'lo-fi': 330}
        base_freq = freq_map.get(genre, 220)
        
        melody = Sine(base_freq).to_audio_segment(duration=2000).apply_gain(-15)
        
        # Add melody at intervals
        melody_track = AudioSegment.silent(duration=total_duration)
        for i in range(0, total_duration, beat_interval * 4):
            if i + len(melody) <= total_duration:
                melody_track = melody_track.overlay(melody, position=i)
        
        return melody_track