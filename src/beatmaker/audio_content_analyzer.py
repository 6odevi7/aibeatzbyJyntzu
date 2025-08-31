import librosa
import numpy as np
from pydub import AudioSegment
import os
from pathlib import Path

class AudioContentAnalyzer:
    def __init__(self):
        self.genre_profiles = {
            'trap': {'tempo_range': (130, 180), 'key_features': ['808', 'hi-hat', 'snare']},
            'boom_bap': {'tempo_range': (80, 110), 'key_features': ['kick', 'snare', 'vinyl']},
            'drill': {'tempo_range': (140, 160), 'key_features': ['slide', '808', 'dark']},
            'lo_fi': {'tempo_range': (70, 90), 'key_features': ['vinyl', 'jazz', 'chill']}
        }
    
    def analyze_audio_content(self, file_path):
        """Analyze audio content for intelligent matching"""
        try:
            y, sr = librosa.load(file_path, duration=30)  # Analyze first 30 seconds
            
            # Extract features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Calculate audio characteristics
            brightness = np.mean(spectral_centroid)
            roughness = np.std(mfcc)
            energy = np.mean(librosa.feature.rms(y=y))
            
            return {
                'tempo': float(tempo),
                'brightness': float(brightness),
                'roughness': float(roughness),
                'energy': float(energy),
                'chroma_profile': chroma.mean(axis=1).tolist(),
                'mfcc_profile': mfcc.mean(axis=1).tolist(),
                'zero_crossing': float(np.mean(zero_crossing_rate))
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def match_content_to_style(self, audio_features, target_style):
        """Match audio content to target style using feature similarity"""
        if not audio_features:
            return 0.0
        
        style_profile = self.genre_profiles.get(target_style, {})
        score = 0.0
        
        # Tempo matching
        tempo_range = style_profile.get('tempo_range', (60, 200))
        if tempo_range[0] <= audio_features['tempo'] <= tempo_range[1]:
            score += 0.3
        
        # Energy matching for style
        if target_style == 'trap' and audio_features['energy'] > 0.1:
            score += 0.2
        elif target_style == 'lo_fi' and audio_features['energy'] < 0.08:
            score += 0.2
        elif target_style == 'drill' and audio_features['energy'] > 0.12:
            score += 0.2
        
        # Brightness matching
        if target_style in ['trap', 'drill'] and audio_features['brightness'] > 2000:
            score += 0.2
        elif target_style in ['boom_bap', 'lo_fi'] and audio_features['brightness'] < 2500:
            score += 0.2
        
        # Roughness for texture
        if target_style == 'lo_fi' and audio_features['roughness'] > 15:
            score += 0.3  # Lo-fi likes texture
        elif target_style == 'trap' and audio_features['roughness'] < 12:
            score += 0.3  # Trap likes clean sounds
        
        return min(score, 1.0)
    
    def find_matching_samples(self, sample_library, target_style, keyword=None, limit=20):
        """Find samples that match target style and keywords through intelligent analysis"""
        matches = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a']
        
        try:
            for root, dirs, files in os.walk(sample_library):
                for file in files:
                    if Path(file).suffix.lower() in audio_extensions:
                        file_path = os.path.join(root, file)
                        filename_lower = file.lower()
                        
                        # Enhanced keyword matching
                        keyword_score = 0.0
                        if keyword:
                            keyword_lower = keyword.lower()
                            # Direct keyword match
                            if keyword_lower in filename_lower:
                                keyword_score += 0.5
                            
                            # Semantic keyword matching
                            keyword_synonyms = self.get_keyword_synonyms(keyword_lower)
                            for synonym in keyword_synonyms:
                                if synonym in filename_lower:
                                    keyword_score += 0.3
                                    break
                            
                            # Style-specific keyword matching
                            style_keywords = self.get_style_keywords(target_style)
                            for style_kw in style_keywords:
                                if style_kw in filename_lower:
                                    keyword_score += 0.2
                                    break
                        else:
                            keyword_score = 0.5  # No keyword filter, include all
                        
                        # Only analyze if keyword matching is promising or no keyword specified
                        if keyword_score > 0.0 or not keyword:
                            # Analyze audio content
                            features = self.analyze_audio_content(file_path)
                            if features:
                                content_score = self.match_content_to_style(features, target_style)
                                
                                # Combine keyword and content scores
                                final_score = (keyword_score * 0.4) + (content_score * 0.6)
                                
                                if final_score > 0.2:  # Lower threshold for more variety
                                    matches.append({
                                        'file_path': file_path,
                                        'match_score': final_score,
                                        'features': features,
                                        'name': Path(file).stem,
                                        'keyword_score': keyword_score,
                                        'content_score': content_score
                                    })
        except Exception as e:
            print(f"Error scanning {sample_library}: {e}")
        
        # Sort by match score and return top matches
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:limit]
    
    def get_keyword_synonyms(self, keyword):
        """Get synonyms for better keyword matching"""
        synonyms = {
            'dark': ['evil', 'sinister', 'shadow', 'black', 'deep', 'heavy'],
            'bright': ['light', 'shiny', 'clean', 'crisp', 'clear', 'sharp'],
            'heavy': ['thick', 'fat', 'big', 'massive', 'huge', 'wide'],
            'soft': ['smooth', 'gentle', 'mellow', 'warm', 'subtle'],
            'hard': ['aggressive', 'punch', 'sharp', 'attack', 'hit'],
            'vocal': ['voice', 'vox', 'acapella', 'chop', 'sample'],
            'piano': ['keys', 'chord', 'melody', 'harmony'],
            'guitar': ['string', 'riff', 'strum', 'pick'],
            'bass': ['low', 'sub', '808', 'bottom', 'deep'],
            'drum': ['perc', 'rhythm', 'beat', 'hit'],
            'synth': ['lead', 'pad', 'wave', 'electronic'],
            'vintage': ['old', 'retro', 'classic', 'analog'],
            'modern': ['new', 'fresh', 'current', 'digital']
        }
        return synonyms.get(keyword, [])
    
    def get_style_keywords(self, style):
        """Get style-specific keywords for better matching"""
        style_keywords = {
            'trap': ['808', 'hi-hat', 'snare', 'roll', 'slide', 'heavy'],
            'boom_bap': ['kick', 'snare', 'vinyl', 'dusty', 'classic', 'old'],
            'drill': ['slide', '808', 'dark', 'aggressive', 'uk', 'drill'],
            'lo_fi': ['vinyl', 'jazz', 'chill', 'dusty', 'warm', 'analog'],
            'hip_hop': ['boom', 'bap', 'classic', 'old', 'school'],
            'gangsta': ['dark', 'heavy', 'aggressive', 'west', 'coast'],
            'phonk': ['memphis', 'dark', 'distorted', 'vintage']
        }
        return style_keywords.get(style, [])