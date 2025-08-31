import os
import numpy as np
from pydub import AudioSegment

class SimpleAudioAnalyzer:
    """Simplified audio analyzer using only pydub - no librosa dependency"""
    
    def analyze_audio_content(self, file_path):
        """Analyze audio file using basic pydub features"""
        try:
            if not os.path.exists(file_path):
                return {}
            
            # Load audio with pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to numpy array for analysis
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = samples.mean(axis=1)  # Convert to mono
            
            # Normalize samples
            if len(samples) > 0:
                samples = samples / np.max(np.abs(samples))
            
            # Basic feature extraction
            features = {}
            
            # Energy (RMS)
            features['energy'] = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
            
            # Zero crossing rate (approximates brightness)
            zero_crossings = np.where(np.diff(np.sign(samples)))[0]
            features['zero_crossing'] = len(zero_crossings) / len(samples) if len(samples) > 0 else 0.0
            
            # Spectral centroid approximation (brightness)
            # Use zero crossing rate as proxy for brightness
            features['brightness'] = features['zero_crossing'] * 10000  # Scale to Hz-like range
            
            # Tempo estimation (very basic)
            # Look for periodic patterns in energy
            if len(samples) > audio.frame_rate:  # At least 1 second
                chunk_size = audio.frame_rate // 10  # 100ms chunks
                energy_chunks = []
                for i in range(0, len(samples) - chunk_size, chunk_size):
                    chunk = samples[i:i + chunk_size]
                    energy_chunks.append(np.sqrt(np.mean(chunk**2)))
                
                # Simple tempo estimation based on energy peaks
                if len(energy_chunks) > 4:
                    # Find peaks in energy
                    energy_array = np.array(energy_chunks)
                    mean_energy = np.mean(energy_array)
                    peaks = []
                    for i in range(1, len(energy_array) - 1):
                        if energy_array[i] > energy_array[i-1] and energy_array[i] > energy_array[i+1] and energy_array[i] > mean_energy:
                            peaks.append(i)
                    
                    if len(peaks) > 1:
                        # Calculate average time between peaks
                        peak_intervals = np.diff(peaks) * 0.1  # Convert to seconds
                        avg_interval = np.mean(peak_intervals)
                        if avg_interval > 0:
                            estimated_bpm = 60 / avg_interval
                            # Clamp to reasonable range
                            features['tempo'] = max(60, min(200, estimated_bpm))
                        else:
                            features['tempo'] = 120.0
                    else:
                        features['tempo'] = 120.0
                else:
                    features['tempo'] = 120.0
            else:
                features['tempo'] = 120.0
            
            # Duration
            features['duration'] = len(audio) / 1000.0  # Convert to seconds
            
            return features
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            # Return default features
            return {
                'energy': 0.5,
                'brightness': 1000,
                'zero_crossing': 0.1,
                'tempo': 120.0,
                'duration': 1.0
            }
    
    def match_content_to_style(self, features, style):
        """Match audio features to style preferences"""
        if not features:
            return 0.5
        
        score = 0.5  # Base score
        
        # Style-specific scoring
        style_lower = style.lower()
        
        if 'trap' in style_lower or 'drill' in style_lower:
            # Prefer high energy, moderate brightness
            if features.get('energy', 0) > 0.3:
                score += 0.2
            if 1000 < features.get('brightness', 0) < 3000:
                score += 0.2
        
        elif 'lo' in style_lower and 'fi' in style_lower:
            # Prefer lower energy, warmer (less bright) sounds
            if features.get('energy', 0) < 0.4:
                score += 0.2
            if features.get('brightness', 0) < 2000:
                score += 0.2
        
        elif 'jazz' in style_lower:
            # Prefer moderate energy, varied brightness
            if 0.2 < features.get('energy', 0) < 0.7:
                score += 0.2
            if features.get('zero_crossing', 0) > 0.05:
                score += 0.1
        
        elif any(word in style_lower for word in ['hip', 'rap', 'boom', 'bap']):
            # Classic hip-hop preferences
            if 0.3 < features.get('energy', 0) < 0.8:
                score += 0.2
            if 1500 < features.get('brightness', 0) < 4000:
                score += 0.1
        
        # Tempo matching
        tempo = features.get('tempo', 120)
        if 'trap' in style_lower and 140 <= tempo <= 180:
            score += 0.1
        elif 'drill' in style_lower and 140 <= tempo <= 160:
            score += 0.1
        elif any(word in style_lower for word in ['boom', 'bap', 'jazz']) and 80 <= tempo <= 110:
            score += 0.1
        elif 'lo' in style_lower and 60 <= tempo <= 90:
            score += 0.1
        
        return min(1.0, max(0.0, score))

# Alias for compatibility
AudioContentAnalyzer = SimpleAudioAnalyzer