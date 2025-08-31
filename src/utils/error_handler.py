import functools
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_execute(fallback_value=None):
    """Decorator that catches all exceptions and returns fallback value"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

def safe_audio_operation(fallback_duration=5000):
    """Decorator for audio operations with audio fallback"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Audio operation failed: {str(e)}")
                from pydub import AudioSegment
                return AudioSegment.silent(duration=fallback_duration)
        return wrapper
    return decorator

class SafeAudioProcessor:
    @staticmethod
    @safe_execute(fallback_value=[])
    def safe_file_scan(directory):
        import os
        from pathlib import Path
        
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a'}
        files = []
        
        if not os.path.exists(directory):
            return []
            
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if Path(filename).suffix.lower() in audio_exts:
                    files.append(os.path.join(root, filename))
        return files
    
    @staticmethod
    @safe_audio_operation(fallback_duration=1000)
    def safe_audio_load(file_path):
        from pydub import AudioSegment
        return AudioSegment.from_file(file_path)
    
    @staticmethod
    @safe_execute(fallback_value={'tempo': 120, 'energy': 0.5, 'brightness': 0.5})
    def safe_audio_analysis(file_path):
        try:
            import librosa
            import numpy as np
            
            y, sr = librosa.load(file_path, duration=10)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness = float(np.mean(spectral_centroid)) / (sr / 2)
            
            return {
                'tempo': float(tempo),
                'energy': min(1.0, energy * 10),
                'brightness': min(1.0, brightness)
            }
        except:
            return {'tempo': 120, 'energy': 0.5, 'brightness': 0.5}