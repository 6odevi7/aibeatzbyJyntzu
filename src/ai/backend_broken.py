from flask import Flask, request, jsonify, Response, send_file, send_from_directory
app = Flask(__name__)
# --- AI Beat Generation Endpoint ---
@app.route('/generate_ai_beat', methods=['POST'])
def generate_ai_beat():
    data = request.get_json()
    ai_model = data.get('aiModel', 'musicgen')
    genre = data.get('genre')
    mood = data.get('mood')
    drumkit = data.get('drumkit')
    tempo = data.get('tempo')
    length = data.get('length')
    eqLow = data.get('eqLow')
    eqMid = data.get('eqMid')
    eqHigh = data.get('eqHigh')
    beatDescription = data.get('beatDescription')
    beatMode = data.get('beatMode')

    # Example: call different logic based on ai_model
    if ai_model == 'musicgen':
        # Call your MusicGen logic here
        beat = {
            'name': f'MusicGen_{genre}_{mood}',
            'type': 'Beat',
            'url': '/static/generated/musicgen_beat.mp3',
            'size': 2048,
            'date': '2025-08-29',
            'samples': 'Kick, Snare'
        }
    elif ai_model == 'spleeter':
        # Call your Spleeter logic here
        beat = {
            'name': f'Spleeter_{genre}_{mood}',
            'type': 'Stems',
            'size': 4096,
            'date': '2025-08-29',
            'samples': 'Kick, Snare',
            'stems': [
                {'name': 'STEM 1 - Kick', 'url': '/static/generated/spleeter_kick.mp3'},
                {'name': 'STEM 2 - Snare', 'url': '/static/generated/spleeter_snare.mp3'}
            ]
        }
    elif ai_model == 'ffmpeg':
        # Call your FFmpeg logic here
        beat = {
            'name': f'FFmpeg_{genre}_{mood}',
            'type': 'Beat',
            'url': '/static/generated/ffmpeg_beat.mp3',
            'size': 1024,
            'date': '2025-08-29',
            'samples': 'Kick, Snare'
        }
    else:
        return jsonify({'error': 'Unknown AI model'}), 400

    return jsonify({'beat': beat})
import os
import sys
import time
import random
from pathlib import Path
try:
    # from spleeter.separator import Separator  # Disabled unresolved import
    Separator = None
except ImportError:
    Separator = None
try:
    import musicgen
    MusicGen = musicgen
except ImportError:
    MusicGen = None

from flask import Flask, Response, request, jsonify, send_file, send_from_directory
app = Flask(__name__)

# --- Global Variables ---
ADMIN_USER = 'admin'
ADMIN_PASS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/admin_pass.txt'))
DRUMKITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../drumkits/unpacked'))
CHOSEN1_PATH = os.environ.get('CHOSEN1_PATH', r'C:/Users/Jintsu/Desktop/Chosen-1')
JINTSU_PATH = os.environ.get('JINTSU_PATH', r'C:/Users/Jintsu/Desktop/JINTSU CATALOG')

# --- Helper Functions ---
def list_audio_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            ext = Path(f).suffix.lower()
            if ext in {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a', '.wma', '.opus'}:
                files.append(os.path.join(dirpath, f))
    return files

def is_drum_sample(filename):
    fname = filename.lower()
    return any(kw in fname for kw in ['kick', 'snare', 'hat', 'perc', 'drum', 'clap', '808', 'rim', 'tom', 'cymbal', 'crash', 'ride', 'shaker'])

def get_admin_password():
    if not os.path.exists(os.path.dirname(ADMIN_PASS_FILE)):
        os.makedirs(os.path.dirname(ADMIN_PASS_FILE), exist_ok=True)
        print(f"[INFO] Created admin password directory: {os.path.dirname(ADMIN_PASS_FILE)}")
    if os.path.exists(ADMIN_PASS_FILE):
        with open(ADMIN_PASS_FILE, 'r') as f:
            pw = f.read().strip()
            print(f"[DEBUG] Read admin password from file: {pw}")
            if pw:
                return pw
    password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))
    with open(ADMIN_PASS_FILE, 'w') as f:
        f.write(password)
    print(f"[INFO] Admin password generated and saved to {ADMIN_PASS_FILE}: {password}")
    return password

# --- New endpoints for frontend compatibility ---
import shutil

def get_itunes_music_folder():
    # Default iTunes music folder for Windows
    return os.path.expanduser(r'~/Music/iTunes/iTunes Media/Music')

def scan_existing_tracks():
    # Scan all sample folders for existing tracks
    all_files = set()
    for folder in [CHOSEN1_PATH, JINTSU_PATH, DRUMKITS_PATH]:
        for f in list_audio_files(folder):
            all_files.add(os.path.basename(f).lower())
    return all_files

@app.route('/import_itunes', methods=['POST'])
def import_itunes():
    itunes_folder = get_itunes_music_folder()
    new_music_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../new music'))
    if not os.path.exists(itunes_folder):
        return jsonify({'status': 'error', 'error': 'iTunes music folder not found.'})
    if not os.path.exists(new_music_folder):
        os.makedirs(new_music_folder, exist_ok=True)
    existing_tracks = scan_existing_tracks()
    imported = []
    skipped = []
    for dirpath, _, filenames in os.walk(itunes_folder):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in {'.mp3', '.wav', '.flac', '.aac', '.m4a'}:
                if fname.lower() in existing_tracks:
                    skipped.append(fname)
                    continue
                src = os.path.join(dirpath, fname)
                dst = os.path.join(new_music_folder, fname)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        imported.append(fname)
                    except Exception:
                        skipped.append(fname)
    return jsonify({'status': 'success', 'imported': imported, 'skipped': skipped, 'count': len(imported)})
import json
ACCOUNTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/accounts.json'))

def load_accounts():
    if os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def save_accounts(accounts):
    with open(ACCOUNTS_FILE, 'w') as f:
        json.dump(accounts, f)

@app.route('/accounts', methods=['GET'])
def get_accounts():
    return jsonify({'accounts': load_accounts()})

@app.route('/accounts', methods=['POST'])
def create_account():
    data = request.json or {}
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username required'}), 400
    accounts = load_accounts()
    if any(acc['username'] == username for acc in accounts):
        return jsonify({'error': 'Username already exists'}), 400
    new_acc = {'username': username, 'banned': False}
    accounts.append(new_acc)
    save_accounts(accounts)
    return jsonify({'status': 'success', 'account': new_acc})

@app.route('/accounts/ban', methods=['POST'])
def ban_account():
    data = request.json or {}
    username = data.get('username')
    accounts = load_accounts()
    for acc in accounts:
        if acc['username'] == username:
            acc['banned'] = True
    save_accounts(accounts)
    return jsonify({'status': 'success'})

@app.route('/accounts/remove', methods=['POST'])
def remove_account():
    data = request.json or {}
    username = data.get('username')
    accounts = [acc for acc in load_accounts() if acc['username'] != username]
    save_accounts(accounts)
    return jsonify({'status': 'success'})

@app.route('/health')
def health():
    import sys
    return jsonify({
        'status': 'ok',
        'file': __file__,
        'cwd': os.getcwd(),
        'python': sys.version,
        'port': request.host.split(':')[-1]
    })

PROPOSALS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/proposals.json'))
FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/feedback.json'))

def generate_drum_beat(genre=None, mood=None, tempo=None, length=None, eq=None, drumkit=None, mode=None, model=None, description=None):
    # All arguments should be passed explicitly, not from params
    if genre is None:
        genre = random.choice(["hip-hop","rap","trap","drill","edm","pop","lo-fi","experimental","metal","custom"])
    if mood is None:
        mood = random.choice(["chill","aggressive","energetic","melancholic","uplifting","sad","dark","happy","epic","intense","dreamy","custom"])
    if drumkit is None:
        drumkit = random.choice(["hip-hop","rap","trap","drill","edm","pop","lo-fi","experimental","metal","custom"])
    if mode is None or mode not in ['ai', 'samples', 'hybrid']:
        mode = 'ai'
    if tempo is None:
        tempo = random.randint(80,160)
    if length is None:
        length = random.randint(8,32)
    # Expanded drum patterns
    drum_patterns = {
        'Hip-Hop': ['kick-snare-hat', 'boom-bap', 'trap-roll', 'syncopated', 'jazzy', 'lofi-dusty'],
        'Rap': ['classic-rap', 'fast-rap', '808-bounce', 'east-coast', 'west-coast', 'southern'],
        'Boom Bap': ['old-school', 'vinyl-crunch', 'hard-hitting', 'swing'],
        'Trap': ['trap-hat', 'snare-roll', 'sub-bass', 'triplet-hats', 'stutter-snare'],
        'Drill': ['drill-hat', 'gliding-808', 'syncopated-snare', 'uk-drill', 'ny-drill'],
        'EDM': ['four-on-the-floor', 'big-room', 'future-bass', 'electro', 'progressive'],
        'Pop': ['pop-groove', 'clap-beat', 'dance-pop', 'synth-pop'],
        'Lo-Fi': ['dusty-kick', 'soft-snare', 'vinyl-hiss', 'chill-hat'],
        'Cloud Rap': ['ambient-hat', 'spacey-snare', 'reverb-kick'],
        'Experimental': ['odd-meter', 'glitch', 'polyrhythm', 'randomized'],
        'Heavy Metal': ['double-kick', 'blast-beat', 'metal-snare'],
        'Metalcore': ['breakdown', 'syncopated-kick', 'china-cymbal'],
        'Custom': ['custom-pattern', 'user-defined'],
        'East Coast': ['boom-bap', 'hard-snare', 'sampled-kick'],
        'West Coast': ['funky-kick', 'laid-back-snare', 'g-funk-hat'],
        'Dirty South': ['crunk', '808-heavy', 'snappy-snare'],
    }
    # Pick a pattern based on genre, mood, and randomization
    genre_key = genre.title() if genre.title() in drum_patterns else 'Hip-Hop'
    patterns = drum_patterns.get(genre_key, drum_patterns['Hip-Hop'])
    # Add mood-based variation
    if mood in ['Aggressive', 'Energetic', 'Intense', 'Epic']:
        patterns += ['double-time', 'extra-kick', 'layered-snare']
    if mood in ['Chill', 'Dreamy', 'Laid Back', 'Melancholic']:
        patterns += ['soft-hat', 'ghost-snare', 'minimal-kick']
    pattern = random.choice(patterns)
    bpm = tempo if tempo else 120
    # ...existing code for beat generation...
# Endpoint: Separate stems from a catalog audio file
# (already fixed above)

# Endpoint: upload a file and find similar matches in generated_beats and user files
from werkzeug.utils import secure_filename

@app.route('/find_similar', methods=['POST'])
def find_similar():
    import librosa, numpy as np
    CHOSEN1_PATH = r'C:/Users/Jintsu/Desktop/Chosen-1'
    JINTSU_PATH = r'C:/Users/Jintsu/Desktop/JINTSU CATALOG'
    uploaded = request.files.get('file')
    if not uploaded:
        return jsonify({'error': 'No file uploaded'}), 400
    filename = secure_filename(uploaded.filename)
    temp_path = os.path.join('generated_beats', f'_temp_{filename}')
    uploaded.save(temp_path)
    # ...existing code...
def is_admin(request):
    auth = request.headers.get('Authorization')
    if not auth:
        return False
    return auth == f"Bearer {get_admin_password()}"

@app.route('/admin_password', methods=['GET'])
def admin_password():
    print(f"[DEBUG] /admin_password endpoint called")
    pw = get_admin_password()
    print(f"[DEBUG] /admin_password returning password: {pw}")
    if not pw:
        print(f"[ERROR] /admin_password: No password found or generated!")
        return jsonify({'error': 'No admin password found.'}), 500
    return jsonify({'admin_user': ADMIN_USER, 'admin_password': pw})

@app.route('/admin_set_password', methods=['POST'])
def admin_set_password():
    data = request.json or {}
    password = data.get('password')
    if not password or len(password) < 8:
        return jsonify({'status': 'error', 'error': 'Password must be at least 8 characters.'})
    with open(ADMIN_PASS_FILE, 'w') as f:
        f.write(password)
    return jsonify({'status': 'success', 'admin_user': ADMIN_USER, 'admin_password': password})

# --- ENFORCE ADMIN FOR PROPOSAL APPROVAL/REVERT ---

@app.route('/puter_review', methods=['POST'])
def puter_review():
    if not is_admin(request):
        return jsonify({'status': 'error', 'error': 'Admin authorization required.'}), 403
    data = request.json or {}
    # --- Improved beat structure ---
    params = request.json or {}
    GENRES = ["hip-hop","rap","trap","drill","edm","pop","lo-fi","experimental","metal","custom"]
    MOODS = ["chill","aggressive","energetic","melancholic","uplifting","sad","dark","happy","epic","intense","dreamy","custom"]
    genre = params.get('genre', '').lower()
    mood = params.get('mood', '').lower()
    drumkit = params.get('drumkit', '').lower()
    params = request.json or {}
    params = request.json or {}
    genre = params.get('genre', '').lower()
    mood = params.get('mood', '').lower()
    drumkit = params.get('drumkit', '').lower()
    # Use explicit arguments, not params
    if mode not in ['ai', 'samples', 'hybrid']:
        mode = 'ai'
    GENRES = ["hip-hop","rap","trap","drill","edm","pop","lo-fi","experimental","metal","custom"]
    MOODS = ["chill","aggressive","energetic","melancholic","uplifting","sad","dark","happy","epic","intense","dreamy","custom"]
    if not genre:
        genre = random.choice(GENRES)
    if not mood:
        mood = random.choice(MOODS)
    if not drumkit:
        drumkit = random.choice(GENRES)
    if tempo is None:
        tempo = random.randint(80,160)
    if length is None:
        length = random.randint(8,32)
    # Use /list_files logic for sample selection
    folders = [CHOSEN1_PATH, JINTSU_PATH, DRUMKITS_PATH, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../new music'))]
    all_sounds = []
    for folder in folders:
        if os.path.exists(folder):
            all_sounds += list_audio_files(folder)
    keywords = [genre, mood, drumkit]
    # If description is available as argument, add to keywords
    if 'description' in locals() and description:
        keywords += description.lower().split()
    keywords = [k for k in keywords if k]
    drumkit_candidates = [s for s in drumkit_candidates if is_drum_sample(s)]
    # Select up to 8 drum samples
    drum_samples = random.sample(drumkit_candidates, min(8, len(drumkit_candidates))) if drumkit_candidates else []
    # Select bass, harmony, melody, fx samples
    def select_role_samples(samples, role_keywords, max_count):
        role_filtered = [s for s in samples if any(k in os.path.basename(s).lower() for k in role_keywords)]
        return random.sample(role_filtered, min(max_count, len(role_filtered))) if role_filtered else []
    bass_samples = select_role_samples(all_sounds, ['808','bass','sub'], 2)
    harmony_samples = select_role_samples(all_sounds, ['pad','chord','harmony','synth','strings'], 2)
    melody_samples = select_role_samples(all_sounds, ['lead','melody','pluck','arp','guitar','piano','sax'], 2)
    fx_samples = select_role_samples(all_sounds, ['fx','voice','vocal','sound','stab','hit'], 2)
    # --- Arrange beat: intro, main loop, outro ---
    from pydub import AudioSegment
    output_dir = os.path.join(os.getcwd(), 'generated_beats')
    os.makedirs(output_dir, exist_ok=True)
    beat_segments = []
    # Intro: 2 bars, use harmony and fx
    for sample in harmony_samples + fx_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:2000] if len(seg) > 2000 else seg
            beat_segments.append(seg)
        except Exception:
            continue
    # Main loop: drums, bass, melody, harmony
    main_loop = []
    for sample in drum_samples + bass_samples + melody_samples + harmony_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:4000] if len(seg) > 4000 else seg
            main_loop.append(seg)
        except Exception:
            continue
    # Stack main loop segments
    if main_loop:
        main = main_loop[0]
        for seg in main_loop[1:]:
            main = main.overlay(seg)
        beat_segments.append(main)
    # Outro: fx and harmony
    for sample in fx_samples + harmony_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:2000] if len(seg) > 2000 else seg
            beat_segments.append(seg)
        except Exception:
            continue
    # Concatenate all segments
    if beat_segments:
        final_beat = beat_segments[0]
        for seg in beat_segments[1:]:
            final_beat += seg
        # Save beat
        beat_name = f"beat_{genre}_{mood}_{drumkit}_{tempo}bpm_{length}bars_{random.randint(1000,9999)}.wav"
        beat_path = os.path.join(output_dir, beat_name)
        final_beat.export(beat_path, format="wav")
        # Return info
        return jsonify({
            'status': 'success',
            'beat_path': beat_path,
            'genre': genre,
            'mood': mood,
            'drumkit': drumkit,
            'tempo': tempo,
            'length': length,
            'samples_used': drum_samples + bass_samples + harmony_samples + melody_samples + fx_samples
        })
    else:
        return jsonify({'status': 'error', 'error': 'No valid samples found for beat.'})
@app.route('/list_files')
def list_files():
    folder = request.args.get('folder')
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    result = {'files': [], 'folders': []}
    if folder == 'generated_beats':
        beats_dir = os.path.abspath(os.path.join(root, 'generated_beats'))
        print(f'[DEBUG] /list_files: folder=generated_beats, beats_dir={beats_dir}, exists={os.path.exists(beats_dir)}')
        if os.path.exists(beats_dir):
            files_found = 0
            for fname in os.listdir(beats_dir):
                fpath = os.path.join(beats_dir, fname)
                if os.path.isfile(fpath):
                    def list_files():
                        # Unified list of all sound files for beatmaking and stem separation
                        folders = [CHOSEN1_PATH, JINTSU_PATH, DRUMKITS_PATH, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../new music'))]
                        all_files = []
                        for folder in folders:
                            if os.path.exists(folder):
                                for f in list_audio_files(folder):
                                    stat = os.stat(f)
                                    all_files.append({
                                        'name': os.path.basename(f),
                                        'path': f,
                                        'size': round(stat.st_size / 1024, 1),
                                        'date': time.ctime(stat.st_mtime),
                                        'url': f"/file?path={f}"
                                    })
                        return jsonify({'files': all_files})
    # Use explicit arguments, not params
    if mode not in ['ai', 'samples', 'hybrid']:
        mode = 'ai'
    GENRES = ["hip-hop","rap","trap","drill","edm","pop","lo-fi","experimental","metal","custom"]
    MOODS = ["chill","aggressive","energetic","melancholic","uplifting","sad","dark","happy","epic","intense","dreamy","custom"]
    if not genre:
        genre = random.choice(GENRES)
    if not mood:
        mood = random.choice(MOODS)
    if not drumkit:
        drumkit = random.choice(GENRES)
    if tempo is None:
        tempo = random.randint(80,160)
    if length is None:
        length = random.randint(8,32)
    # Recursively list all audio files in sample paths
    chosen1_sounds = list_audio_files(CHOSEN1_PATH)
    jintsu_sounds = list_audio_files(JINTSU_PATH)
    drumkit_sounds = list_audio_files(DRUMKITS_PATH)
    all_sounds = chosen1_sounds + jintsu_sounds + drumkit_sounds
    keywords = [genre, mood, drumkit]
    if description:
        keywords += description.lower().split()
    keywords = [k for k in keywords if k]
    def filter_samples(samples, keywords):
        filtered = [s for s in samples if any(k in os.path.basename(s).lower() for k in keywords)]
        return filtered if filtered else samples
    drumkit_candidates = filter_samples(drumkit_sounds, keywords)
    drumkit_candidates = [s for s in drumkit_candidates if is_drum_sample(s)]
    drum_samples = random.sample(drumkit_candidates, min(8, len(drumkit_candidates))) if drumkit_candidates else []
    def select_role_samples(samples, role_keywords, max_count):
        role_filtered = [s for s in samples if any(k in os.path.basename(s).lower() for k in role_keywords)]
        return random.sample(role_filtered, min(max_count, len(role_filtered))) if role_filtered else []
    bass_samples = select_role_samples(all_sounds, ['808','bass','sub'], 2)
    harmony_samples = select_role_samples(all_sounds, ['pad','chord','harmony','synth','strings'], 2)
    melody_samples = select_role_samples(all_sounds, ['lead','melody','pluck','arp','guitar','piano','sax'], 2)
    fx_samples = select_role_samples(all_sounds, ['fx','voice','vocal','sound','stab','hit'], 2)
    # Log sample counts for debugging
    print(f"[DEBUG] chosen1_sounds: {len(chosen1_sounds)} files")
    print(f"[DEBUG] jintsu_sounds: {len(jintsu_sounds)} files")
    print(f"[DEBUG] drumkit_sounds: {len(drumkit_sounds)} files")
    print(f"[DEBUG] drumkit_candidates: {len(drumkit_candidates)} files")
    print(f"[DEBUG] drum_samples: {len(drum_samples)} files")
    print(f"[DEBUG] bass_samples: {len(bass_samples)} files")
    print(f"[DEBUG] harmony_samples: {len(harmony_samples)} files")
    print(f"[DEBUG] melody_samples: {len(melody_samples)} files")
    print(f"[DEBUG] fx_samples: {len(fx_samples)} files")
    from pydub import AudioSegment
    output_dir = os.path.join(os.getcwd(), 'generated_beats')
    os.makedirs(output_dir, exist_ok=True)
    beat_segments = []
    # Intro: 2 bars, use harmony and fx
    for sample in harmony_samples + fx_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:2000] if len(seg) > 2000 else seg
            beat_segments.append(seg)
        except Exception:
            continue
    # Always fulfill drum track: overlay all drum samples for a steady drum layer
    if not drum_samples:
        print('[ERROR] No drum samples found, cannot fulfill drum track.')
        return jsonify({'status': 'error', 'error': 'No drum samples found. Please add drum samples to your folders.'})
    drum_layer = None
    for sample in drum_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:4000] if len(seg) > 4000 else seg
            if drum_layer is None:
                drum_layer = seg
            else:
                drum_layer = drum_layer.overlay(seg)
        except Exception:
            continue
    # Overlay bass, melody, harmony on top of steady drum layer
    main = drum_layer
    for sample in bass_samples + melody_samples + harmony_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:4000] if len(seg) > 4000 else seg
            main = main.overlay(seg)
        except Exception:
            continue
    if main:
        beat_segments.append(main)
    # Outro: fx and harmony
    for sample in fx_samples + harmony_samples:
        try:
            seg = AudioSegment.from_file(sample)
            seg = seg[:2000] if len(seg) > 2000 else seg
            beat_segments.append(seg)
        except Exception:
            continue
    # Concatenate all segments
    if beat_segments:
        final_beat = beat_segments[0]
        for seg in beat_segments[1:]:
            final_beat += seg
        # Save beat
        beat_name = f"beat_{genre}_{mood}_{drumkit}_{tempo}bpm_{length}bars_{random.randint(1000,9999)}.wav"
        beat_path = os.path.join(output_dir, beat_name)
        final_beat.export(beat_path, format="wav")
        # Return info with beat metadata for media section
        beat_info = {
            'name': beat_name,
            'path': beat_path,
            'genre': genre,
            'mood': mood,
            'drumkit': drumkit,
            'tempo': tempo,
            'length': length,
            'samples_used': drum_samples + bass_samples + harmony_samples + melody_samples + fx_samples
        }
        return jsonify({'status': 'success', 'beat': beat_info})
    else:
        return jsonify({'status': 'error', 'error': 'No valid samples found for beat.'})

# --- Endpoint: List available drumkits ---
@app.route('/list_drumkits')
def list_drumkits():
    drumkits_dir = DRUMKITS_PATH
    drumkits = []
    if os.path.exists(drumkits_dir):
        for fname in os.listdir(drumkits_dir):
            fpath = os.path.join(drumkits_dir, fname)
            if os.path.isdir(fpath) or os.path.isfile(fpath):
                drumkits.append({
                    'name': fname,
                    'path': fpath,
                    'type': 'dir' if os.path.isdir(fpath) else 'file'
                })
    return jsonify({'drumkits': drumkits})

# --- Endpoint: Beat generation progress (stub) ---
@app.route('/generate_beat_progress')
def generate_beat_progress():
    # Stub: always return 100% for now
    return jsonify({'progress': 100, 'status': 'complete'})

# --- Flask app startup ---
# Ensure this is at the end of the file, after 'app' is defined

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
import sys
import json
import time

def main():
    print('Python backend started', file=sys.stderr)
    if len(sys.argv) < 2:
        print(json.dumps({'status': 'error', 'error': 'No input params'}))
        return
    try:
        params = json.loads(sys.argv[1])
        print(f'Received params: {params}', file=sys.stderr)
        # Simulate beat generation
        time.sleep(2)
        # Simulate drumkit loading
        drumkits = ['808', 'Acoustic', 'Trap', 'Lo-Fi']
        print(f'Drumkits available: {drumkits}', file=sys.stderr)
        # Return a dummy beat result
        result = {
            'status': 'success',
            'beat': 'beat1.mp3',
            'title': 'Generated Beat',
            'used_samples': ['Kick', 'Snare'],
            'size': 1234,
            'drumkits': drumkits
        }
        print(json.dumps(result))
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        print(json.dumps({'status': 'error', 'error': str(e)}))

if __name__ == '__main__':
    main()
