import random
import numpy as np
from pydub import AudioSegment
from audio_content_analyzer import AudioContentAnalyzer
from sample_manager import SampleManager
import os
import sys
import argparse
import json

class CatalogBeatGenerator:
    def __init__(self, sample_directories):
        self.analyzer = AudioContentAnalyzer()
        self.sample_manager = SampleManager()
        self.sample_directories = sample_directories
        self.generation_results = {}
        
        # Scan sample directories
        self._scan_sample_directories()
    
    def _scan_sample_directories(self):
        """Scan all sample directories and populate database"""
        print(f"Scanning {len(self.sample_directories)} sample directories...")
        for directory in self.sample_directories:
            if os.path.exists(directory):
                print(f"Scanning: {directory}")
                self._scan_directory_recursive(directory)
            else:
                print(f"Directory not found: {directory}")
                
    def _scan_directory_recursive(self, directory):
        """Recursively scan directory for audio files"""
        sample_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.aiff', '.flac')):
                    file_path = os.path.join(root, file)
                    try:
                        # Quick check if already in database
                        existing = self.sample_manager.search_samples(query=file_path)
                        if not existing:
                            # Analyze audio content
                            features = self.analyzer.analyze_audio_content(file_path)
                            
                            # Add to database
                            self.sample_manager.add_sample(
                                file_path=file_path
                            )
                            sample_count += 1
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
        print(f"Added {sample_count} new samples from {directory}")
    
    def generate_beat(self, genre, style='modern', tempo=120, bars=32, keywords=None, mood='neutral', complexity='medium'):
        """Generate beat using entire sample catalog with audio analysis"""
        keywords = keywords or []
        
        # Get samples from catalog using intelligent selection
        selected_samples = self._select_samples_from_catalog(
            genre=genre,
            keywords=keywords,
            mood=mood,
            complexity=complexity,
            target_count=15
        )
        
        # Apply style-specific processing based on audio analysis
        processed_samples = self._apply_style_processing(selected_samples, style, genre)
        
        # Create beat with audio analysis-driven arrangement
        beat = self._create_beat_with_analysis(processed_samples, genre, style, tempo, bars)
        
        return beat
    
    def _select_samples_from_catalog(self, genre, keywords, mood, complexity, target_count):
        """Select samples from entire catalog using audio analysis"""
        # Build search keywords
        all_keywords = [genre] + keywords + [mood]
        
        # Search for samples matching criteria from entire catalog
        candidate_samples = self.sample_manager.search_samples(
            query=' '.join(all_keywords),
            genre=genre,
            limit=target_count * 5
        )
        
        # If not enough matches, broaden search
        if len(candidate_samples) < target_count:
            print(f"Broadening search - found {len(candidate_samples)} candidates")
            candidate_samples.extend(
                self.sample_manager.search_samples(
                    query=genre,
                    limit=target_count * 3
                )
            )
            
        # Remove duplicates
        seen_paths = set()
        unique_candidates = []
        for sample in candidate_samples:
            if sample['file_path'] not in seen_paths:
                unique_candidates.append(sample)
                seen_paths.add(sample['file_path'])
        candidate_samples = unique_candidates
        
        # Score samples based on audio content analysis
        scored_samples = []
        for sample in candidate_samples:
            # Analyze features on demand
            try:
                features = self.analyzer.analyze_audio_content(sample['file_path'])
            except:
                features = {}
            match_score = self.analyzer.match_content_to_style(features, genre) if features else 0.5
            
            # Boost score based on mood and complexity
            if mood == 'dark' and features.get('brightness', 0.5) < 0.4:
                match_score += 0.2
            elif mood == 'bright' and features.get('brightness', 0.5) > 0.6:
                match_score += 0.2
                
            scored_samples.append({
                'file': sample['file_path'],
                'features': features,
                'score': match_score
            })
        
        # Sort by score and select top samples
        scored_samples.sort(key=lambda x: x['score'], reverse=True)
        selected_samples = scored_samples[:target_count]
        
        # If we don't have enough samples, add some from catalog
        if len(selected_samples) < target_count:
            print(f"Need {target_count - len(selected_samples)} more samples")
            additional_samples = self.sample_manager.search_samples(
                limit=target_count - len(selected_samples)
            )
            for sample in additional_samples:
                selected_samples.append({
                    'file': sample['file_path'],
                    'features': {},
                    'score': 0.5
                })
        
        print(f"Selected {len(selected_samples)} samples from catalog")
        return selected_samples
    
    def _apply_style_processing(self, samples, style, genre):
        """Apply style-specific processing based on audio analysis"""
        processed = []
        
        # Analyze overall audio characteristics
        avg_brightness = np.mean([s['features'].get('brightness', 0.5) for s in samples])
        avg_energy = np.mean([s['features'].get('energy', 0.5) for s in samples])
        
        audio_features = {
            'brightness': avg_brightness,
            'energy': avg_energy
        }
        
        for sample in samples:
            sample_copy = sample.copy()
            
            # Apply style-specific processing
            if style == 'old_school' or 'old' in style:
                # Darken and add character
                brightness_reduction = 0.3 + (avg_brightness * 0.4)
                sample_copy['processing'] = {
                    'vintage_warmth': True,
                    'brightness_reduction': brightness_reduction,
                    'warmth_boost': 0.4 + (avg_energy * 0.3),
                    'style': 'darkened_vintage'
                }
            else:
                # Modern, crisp sound
                brightness_boost = 0.2 + ((1.0 - avg_brightness) * 0.4)
                sample_copy['processing'] = {
                    'digital_clarity': True,
                    'brightness_boost': brightness_boost,
                    'clarity_enhancement': 0.3 + (avg_energy * 0.4),
                    'style': 'brightened_modern'
                }
            
            processed.append(sample_copy)
        
        # Store generation results
        self.generation_results = {
            'samples_used': [s['file'] for s in samples],
            'audio_features': audio_features,
            'style_processing': f"{style} era processing applied with audio analysis",
            'genre_characteristics': genre
        }
        
        return processed
    
    def _create_beat_with_analysis(self, samples, genre, style, tempo, bars):
        """Create beat using audio analysis to drive arrangement"""
        # Calculate beat length
        beat_length_ms = (60000 / tempo) * 4 * bars
        beat = AudioSegment.silent(duration=int(beat_length_ms))
        
        # Categorize samples by audio content
        categorized = self._categorize_by_audio_content(samples)
        
        # Create genre-specific patterns
        patterns = self._get_genre_patterns(genre)
        
        # Build beat with audio-driven arrangement
        beat = self._build_audio_driven_beat(beat, categorized, patterns, tempo, bars)
        
        return beat
    
    def _categorize_by_audio_content(self, samples):
        """Categorize samples by audio content analysis"""
        categories = {'kick': [], 'snare': [], 'hihat': [], 'perc': [], 'melody': [], 'bass': [], '808': []}
        
        for sample in samples:
            features = sample['features']
            category = self._classify_by_audio_features(features, sample['file'])
            categories[category].append(sample)
        
        return categories
    
    def _classify_by_audio_features(self, features, file_path):
        """Classify sample type based on audio features"""
        filename = os.path.basename(file_path).lower()
        
        # 808 detection first
        if any(word in filename for word in ['808', 'eight', 'sub', 'subbass']):
            return '808'
        
        # Filename-based classification
        if any(word in filename for word in ['kick', 'bd', 'bassdrum']):
            return 'kick'
        elif any(word in filename for word in ['snare', 'sn', 'clap']):
            return 'snare'
        elif any(word in filename for word in ['hat', 'hh', 'hihat']):
            return 'hihat'
        elif any(word in filename for word in ['perc', 'shaker', 'rim']):
            return 'perc'
        elif any(word in filename for word in ['bass']):
            return 'bass'
        
        # Feature-based classification with 808 detection
        if features.get('energy', 0) > 0.2 and features.get('brightness', 1000) < 800:
            return '808'
        elif features.get('energy', 0) > 0.15 and features.get('brightness', 1000) < 1500:
            return 'kick'
        elif features.get('energy', 0) > 0.12 and features.get('brightness', 1000) > 3000:
            return 'snare'
        elif features.get('brightness', 1000) > 5000 and features.get('zero_crossing', 0) > 0.1:
            return 'hihat'
        elif features.get('brightness', 1000) < 1200:
            return 'bass'
        else:
            return 'melody'
    
    def _get_genre_patterns(self, genre):
        """Get genre-specific patterns"""
        patterns = {
            'trap': {
                'kick': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                'hihat': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                '808': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
            },
            'drill': {
                'kick': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                '808': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
            },
            'hip-hop': {
                'kick': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                'snare': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                '808': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            }
        }
        
        # Default to hip-hop pattern if genre not found
        return patterns.get(genre.lower(), patterns['hip-hop'])
    
    def _build_audio_driven_beat(self, beat, categorized, patterns, tempo, bars):
        """Build beat using audio analysis to drive sample selection"""
        step_length_ms = len(beat) / (16 * bars)
        
        # Add drums with audio-driven selection
        for pattern_type in ['kick', 'snare', 'hihat', '808']:
            if pattern_type in patterns and categorized[pattern_type]:
                pattern = patterns[pattern_type]
                samples = categorized[pattern_type][:3]  # Use top 3 matches
                
                for bar in range(bars):
                    for step, hit in enumerate(pattern):
                        if hit:
                            # Select sample based on audio characteristics
                            selected_sample = self._select_sample_by_audio_context(
                                samples, bar, step, bars, pattern_type
                            )
                            
                            if selected_sample:
                                audio = AudioSegment.from_file(selected_sample['file'])
                                
                                # Apply audio-driven processing
                                audio = self._apply_audio_driven_processing(
                                    audio, selected_sample, pattern_type, bar, bars
                                )
                                
                                position = int((bar * 16 + step) * step_length_ms)
                                beat = beat.overlay(audio, position=position)
        
        # Add melodic elements based on audio analysis
        beat = self._add_melodic_elements_with_analysis(beat, categorized, tempo, bars)
        
        return beat
    
    def _select_sample_by_audio_context(self, samples, bar, step, total_bars, sample_type):
        """Select sample based on audio context and musical position"""
        if not samples:
            return None
        
        # Use different samples for different sections based on audio characteristics
        if bar < total_bars * 0.25:  # Intro - use softer samples
            return min(samples, key=lambda s: s['features'].get('energy', 0.5))
        elif bar >= total_bars * 0.75:  # Outro - use varied samples
            return samples[step % len(samples)]
        else:  # Main section - use best matches with variation
            return samples[random.randint(0, min(1, len(samples)-1))]
    
    def _apply_audio_driven_processing(self, audio, sample, sample_type, bar, total_bars):
        """Apply processing based on audio analysis"""
        processing = sample.get('processing', {})
        
        # Apply style-specific processing
        if processing.get('style') == 'darkened_vintage':
            # Darken the sound
            audio = audio.low_pass_filter(8000)
            audio = audio + random.uniform(-2, 2)  # Slight volume variation
        elif processing.get('style') == 'brightened_modern':
            # Brighten the sound
            audio = audio.high_pass_filter(80)
            audio = audio + random.uniform(0, 3)  # Slight boost
        
        # 808 specific processing
        if sample_type == '808':
            # Add pitch variation for musicality
            if random.random() < 0.7:
                pitch_shift = random.choice([-5, -3, -2, 0, 2, 3, 5])
                audio = self._pitch_shift(audio, pitch_shift)
            
            # Boost 808 volume
            audio = audio + random.uniform(3, 7)
        
        return audio
    
    def _add_melodic_elements_with_analysis(self, beat, categorized, tempo, bars):
        """Add melodic elements using audio analysis"""
        beat_length = len(beat)
        
        # Add 808/bass line with musical intelligence
        if categorized['808'] or categorized['bass']:
            bass_samples = categorized['808'] or categorized['bass']
            
            # Create musical bassline positions
            positions = []
            step_length = beat_length / (16 * bars)
            
            for bar in range(bars):
                base_pos = int(bar * 16 * step_length)
                positions.extend([
                    base_pos + int(0 * step_length),   # Downbeat
                    base_pos + int(6 * step_length),   # Syncopated
                    base_pos + int(10 * step_length),  # Off-beat
                    base_pos + int(14 * step_length)   # Leading to next bar
                ])
            
            for i, pos in enumerate(positions):
                if pos < beat_length:
                    bass_sample = bass_samples[i % len(bass_samples)]
                    bass_audio = AudioSegment.from_file(bass_sample['file'])
                    
                    # Add musical pitch variation
                    if i % 4 != 0:  # Not on downbeats
                        pitch_shift = random.choice([-3, -2, 2, 3])
                        bass_audio = self._pitch_shift(bass_audio, pitch_shift)
                    
                    beat = beat.overlay(bass_audio, position=pos)
        
        # Add melody layers
        if categorized['melody']:
            melody_samples = categorized['melody'][:2]
            
            for i, sample in enumerate(melody_samples):
                pos = int(beat_length * (0.25 + i * 0.3))  # Spaced out melodies
                if pos < beat_length:
                    melody_audio = AudioSegment.from_file(sample['file'])
                    melody_audio = melody_audio - random.uniform(4, 8)  # Reduce volume
                    beat = beat.overlay(melody_audio, position=pos)
        
        return beat
    
    def _pitch_shift(self, audio, semitones):
        """Simple pitch shifting"""
        if semitones == 0:
            return audio
        
        rate_change = 2 ** (semitones / 12.0)
        if rate_change > 1:
            return audio[::int(rate_change)] if len(audio) > 100 else audio
        else:
            return audio + audio[:int(len(audio) * (1 - rate_change))]

def main():
    parser = argparse.ArgumentParser(description='Generate beats using full sample catalog with audio analysis')
    parser.add_argument('--genre', required=True, help='Beat genre')
    parser.add_argument('--style', default='modern', help='Beat style (modern/old_school)')
    parser.add_argument('--tempo', type=int, default=120, help='Beat tempo')
    parser.add_argument('--bars', type=int, default=32, help='Beat length in bars')
    parser.add_argument('--keyword', default='', help='Style keywords')
    parser.add_argument('--eq_low', type=float, default=0, help='EQ low')
    parser.add_argument('--eq_mid', type=float, default=0, help='EQ mid')
    parser.add_argument('--eq_high', type=float, default=0, help='EQ high')
    parser.add_argument('--description', default='', help='Beat description')
    parser.add_argument('--mode', default='creative', help='Generation mode')
    parser.add_argument('--audio_analysis', type=bool, default=True, help='Use audio analysis')
    parser.add_argument('--mood', default='neutral', help='Beat mood')
    parser.add_argument('--complexity', default='medium', help='Beat complexity')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--chosen1_path', required=True, help='Chosen-1 samples path')
    parser.add_argument('--jintsu_path', required=True, help='JINTSU catalog path')
    parser.add_argument('--drumkits_path', required=True, help='Drumkits path')
    
    args = parser.parse_args()
    
    # Sample directories from arguments
    sample_dirs = [args.chosen1_path, args.jintsu_path, args.drumkits_path]
    
    generator = CatalogBeatGenerator(sample_dirs)
    
    # Parse keywords
    keywords = [k.strip() for k in args.keyword.split(',') if k.strip()] if args.keyword else []
    
    # Generate beat using catalog
    beat = generator.generate_beat(
        genre=args.genre,
        style=args.style,
        tempo=args.tempo,
        bars=args.bars,
        keywords=keywords,
        mood=args.mood,
        complexity=args.complexity
    )
    
    # Output results as JSON for the frontend
    result = {
        'status': 'success',
        'samples_used': generator.generation_results.get('samples_used', []),
        'audio_features': generator.generation_results.get('audio_features', {}),
        'style_processing': generator.generation_results.get('style_processing', ''),
        'beat_data': {
            'samples': len(generator.generation_results.get('samples_used', [])),
            'tempo': args.tempo,
            'bars': args.bars,
            'genre': args.genre,
            'duration_seconds': len(beat) / 1000.0
        }
    }
    
    print(json.dumps(result))
    
    # Save beat file
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        beat.export(args.output, format="wav")
        print(f"Beat saved to: {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving beat: {e}", file=sys.stderr)
        # Create placeholder file
        with open(args.output, 'w') as f:
            f.write('# Generated beat placeholder\n')

if __name__ == "__main__":
    main()