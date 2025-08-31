import os
import sys
import json
import argparse
import random
from pydub import AudioSegment
from pydub.generators import Sine, Square

def generate_simple_beat(genre, tempo, bars, output_path):
    """Generate a simple beat using synthesized sounds"""
    
    # Calculate beat length
    beat_length_ms = (60000 / tempo) * 4 * bars
    beat = AudioSegment.silent(duration=int(beat_length_ms))
    
    # Step length (16th notes)
    step_length_ms = beat_length_ms / (16 * bars)
    
    # Genre patterns
    patterns = {
        'trap': {'kick': [1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0], 'snare': [0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1], 'hihat': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]},
        'hip-hop': {'kick': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], 'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0], 'hihat': [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]},
        'drill': {'kick': [1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0], 'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0], 'hihat': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}
    }
    
    # Default to hip-hop if genre not found
    pattern = patterns.get(genre.lower(), patterns['hip-hop'])
    
    # Generate sounds
    kick = Sine(60).to_audio_segment(duration=200).fade_out(150)
    snare = Square(200).to_audio_segment(duration=150).fade_out(100) + AudioSegment.from_mono_audiosegments(
        *[Square(random.randint(1000, 3000)).to_audio_segment(duration=50) for _ in range(3)]
    )
    hihat = Square(8000).to_audio_segment(duration=50).fade_out(40)
    
    # Add drums
    for bar in range(bars):
        for step, hit in enumerate(pattern['kick']):
            if hit:
                pos = int((bar * 16 + step) * step_length_ms)
                beat = beat.overlay(kick, position=pos)
        
        for step, hit in enumerate(pattern['snare']):
            if hit:
                pos = int((bar * 16 + step) * step_length_ms)
                beat = beat.overlay(snare, position=pos)
        
        for step, hit in enumerate(pattern['hihat']):
            if hit:
                pos = int((bar * 16 + step) * step_length_ms)
                beat = beat.overlay(hihat, position=pos)
    
    # Normalize and export
    beat = beat.normalize()
    beat.export(output_path, format="wav")
    
    return beat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genre', required=True)
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--bars', type=int, default=32)
    parser.add_argument('--output', required=True)
    
    # Accept all other args to avoid errors
    parser.add_argument('--style', default='modern')
    parser.add_argument('--keyword', default='')
    parser.add_argument('--eq_low', type=float, default=0)
    parser.add_argument('--eq_mid', type=float, default=0)
    parser.add_argument('--eq_high', type=float, default=0)
    parser.add_argument('--description', default='')
    parser.add_argument('--mode', default='creative')
    parser.add_argument('--audio_analysis', default=True)
    parser.add_argument('--mood', default='neutral')
    parser.add_argument('--complexity', default='medium')
    parser.add_argument('--chosen1_path', default='')
    parser.add_argument('--jintsu_path', default='')
    parser.add_argument('--drumkits_path', default='')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        beat = generate_simple_beat(args.genre, args.tempo, args.bars, args.output)
        
        result = {
            'status': 'success',
            'samples_used': 'Synthesized drums',
            'audio_features': {'tempo': args.tempo, 'bars': args.bars},
            'style_processing': f'{args.genre} pattern applied',
            'beat_data': {
                'samples': 3,
                'tempo': args.tempo,
                'bars': args.bars,
                'genre': args.genre,
                'duration_seconds': len(beat) / 1000.0
            }
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'status': 'error', 'error': str(e)}))

if __name__ == "__main__":
    main()