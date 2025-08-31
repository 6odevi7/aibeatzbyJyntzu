import os
import random
import numpy as np
from pydub import AudioSegment

class SafeAudioAnalyzer:
    def __init__(self):
        pass
    
    def analyze_audio_content(self, file_path):
        """Analyze audio content safely without crashes"""
        try:
            # Load audio with error handling
            audio = AudioSegment.from_file(file_path)
            
            # Convert to numpy array safely
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)  # Convert to mono
            
            # Normalize
            if len(samples) > 0:
                samples = samples.astype(np.float32) / np.max(np.abs(samples))
            else:
                return None
            
            # Calculate basic features safely
            features = {}
            
            # Energy (RMS)
            features['energy'] = float(np.sqrt(np.mean(samples**2)))
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(samples)))[0]
            features['zero_crossing'] = len(zero_crossings) / len(samples)
            
            # Spectral centroid (brightness approximation)
            # Simple frequency analysis
            fft = np.fft.fft(samples[:min(len(samples), 4096)])
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(fft), 1/audio.frame_rate)
            
            # Calculate centroid
            if np.sum(magnitude) > 0:
                features['brightness'] = float(np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2]))
            else:
                features['brightness'] = 1000.0
            
            # Tempo estimation (simple)
            features['tempo'] = self.estimate_tempo(samples, audio.frame_rate)
            
            return features
            
        except Exception as e:
            print(f"Audio analysis failed for {file_path}: {e}")
            # Return default features
            return {
                'energy': 0.5,
                'zero_crossing': 0.1,
                'brightness': 2000.0,
                'tempo': 120.0
            }
    
    def estimate_tempo(self, samples, sample_rate):
        """Simple tempo estimation"""
        try:
            # Simple onset detection
            if len(samples) < 1024:
                return 120.0
            
            # Calculate energy in windows
            window_size = 1024
            hop_size = 512
            energy = []
            
            for i in range(0, len(samples) - window_size, hop_size):
                window = samples[i:i + window_size]
                energy.append(np.sum(window**2))
            
            if len(energy) < 10:
                return 120.0
            
            # Find peaks in energy
            energy = np.array(energy)
            diff = np.diff(energy)
            peaks = []
            
            for i in range(1, len(diff)):
                if diff[i-1] > 0 and diff[i] < 0:
                    peaks.append(i)
            
            if len(peaks) < 2:
                return 120.0
            
            # Calculate average interval between peaks
            intervals = np.diff(peaks)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals) * hop_size / sample_rate
                if avg_interval > 0:
                    bpm = 60.0 / avg_interval
                    # Clamp to reasonable range
                    return max(60.0, min(200.0, bpm))
            
            return 120.0
            
        except:
            return 120.0
    
    def match_content_to_style(self, features, style):
        """Match audio features to style"""
        try:
            score = 0.5  # Base score
            
            # Style-specific matching
            if style in ['trap', 'drill']:
                # Prefer high energy, bright sounds
                if features['energy'] > 0.3:
                    score += 0.2
                if features['brightness'] > 3000:
                    score += 0.1
            elif style in ['boom', 'bap', 'jazz']:
                # Prefer warmer, less bright sounds
                if features['brightness'] < 2000:
                    score += 0.2
                if features['energy'] > 0.2 and features['energy'] < 0.7:
                    score += 0.1
            elif style in ['lo', 'fi', 'chill']:
                # Prefer lower energy
                if features['energy'] < 0.4:
                    score += 0.2
                if features['brightness'] < 1500:
                    score += 0.1
            
            # Add some randomness for variety
            score += random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, score))
            
        except:
            return 0.5
    
    def classify_sample_type(self, features, file_path):
        """Classify sample type based on audio features"""
        try:
            filename = os.path.basename(file_path).lower()
            
            # Filename hints first
            if any(word in filename for word in ['808', 'sub', 'bass']):
                return '808'
            elif any(word in filename for word in ['kick', 'bd']):
                return 'kick'
            elif any(word in filename for word in ['snare', 'clap']):
                return 'snare'
            elif any(word in filename for word in ['hat', 'hh']):
                return 'hihat'
            elif any(word in filename for word in ['perc', 'shaker']):
                return 'perc'
            
            # Audio feature classification
            energy = features.get('energy', 0.5)
            brightness = features.get('brightness', 2000)
            zero_crossing = features.get('zero_crossing', 0.1)
            
            # 808/Bass: High energy, low brightness
            if energy > 0.3 and brightness < 800:
                return '808'
            # Kick: High energy, low-mid brightness
            elif energy > 0.4 and brightness < 1500:
                return 'kick'
            # Snare: Medium-high energy, high brightness
            elif energy > 0.3 and brightness > 2500:
                return 'snare'
            # Hihat: Low energy, very high brightness, high zero crossing
            elif brightness > 4000 and zero_crossing > 0.15:
                return 'hihat'
            # Bass: Low brightness
            elif brightness < 1000:
                return 'bass'
            # Everything else is melody
            else:
                return 'melody'
                
        except:
            return 'melody'
    
    def find_matching_samples(self, path, style, keyword, limit=10):
        """Find samples that match style and keyword with audio analysis"""
        try:
            from pathlib import Path
            
            # Get audio files
            audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a'}
            files = []
            
            for root, dirs, filenames in os.walk(path):
                for filename in filenames:
                    if Path(filename).suffix.lower() in audio_exts:
                        files.append(os.path.join(root, filename))
            
            if not files:
                return []
            
            # Limit files to prevent long processing
            if len(files) > 100:
                files = random.sample(files, 100)
            
            matches = []
            
            for file_path in files:
                try:
                    # Keyword filtering
                    if keyword and keyword.lower() not in os.path.basename(file_path).lower():
                        continue
                    
                    # Analyze audio content
                    features = self.analyze_audio_content(file_path)
                    if not features:
                        continue
                    
                    # Calculate match score
                    match_score = self.match_content_to_style(features, style)
                    
                    # Classify sample type
                    sample_type = self.classify_sample_type(features, file_path)
                    
                    matches.append({
                        'file_path': file_path,
                        'match_score': match_score,
                        'features': features,
                        'type': sample_type
                    })
                    
                except Exception as e:
                    print(f"Failed to analyze {file_path}: {e}")
                    continue
            
            # Sort by match score and return top matches
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            return matches[:limit]
            
        except Exception as e:
            print(f"Sample matching failed for {path}: {e}")
            return []