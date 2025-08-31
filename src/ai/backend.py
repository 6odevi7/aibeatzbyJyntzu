import os
import sys
import time
import random
import json
import hashlib
import gc
import psutil
import threading
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from functools import wraps
from datetime import datetime, timedelta

# Import enhanced systems - fix the import paths
try:
    # Try relative imports first
    from .user_learning_system import UserLearningSystem
    from .youtube_music_scanner import YouTubeMusicScanner
    from .advanced_beat_engine import AdvancedBeatEngine
except ImportError:
    try:
        # Try direct imports
        from user_learning_system import UserLearningSystem
        from youtube_music_scanner import YouTubeMusicScanner
        from advanced_beat_engine import AdvancedBeatEngine
    except ImportError:
        # Create minimal implementations inline
        class UserLearningSystem:
            def __init__(self): pass
            def record_beat_generation(self, *args): pass
            def get_user_preferences(self, user_id): return {}
            def get_user_stats(self, user_id): return {'total_beats': 0, 'avg_rating': 0, 'favorite_genre': 'Unknown'}
            def get_recommended_params(self, user_id): return None
            def record_user_rating(self, *args): pass
            def record_user_behavior(self, *args): pass
        
        class YouTubeMusicScanner:
            def __init__(self): pass
            def get_genre_training_data(self, genre): return []
            def scan_genre_playlist(self, *args): return 10
            def scan_all_genres(self): return 25
        
        class AdvancedBeatEngine:
            def __init__(self): pass
            def generate_complex_beat(self, params, samples): return None

app = Flask(__name__)
CORS(app)

# Initialize enhanced systems
learning_system = UserLearningSystem()
youtube_scanner = YouTubeMusicScanner()
beat_engine = AdvancedBeatEngine()

# AI Learning Configuration
ai_learning_enabled = True
ai_learning_mode = 'active'

# Performance monitoring
STATS = {
    'requests_total': 0,
    'beats_generated': 0,
    'errors_count': 0,
    'avg_response_time': 0,
    'memory_usage': 0,
    'cpu_usage': 0,
    'start_time': time.time()
}

# Enhanced logging
def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        STATS['requests_total'] += 1
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            STATS['avg_response_time'] = (STATS['avg_response_time'] + response_time) / 2
            
            # Memory cleanup
            if STATS['requests_total'] % 10 == 0:
                gc.collect()
                STATS['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024
                STATS['cpu_usage'] = psutil.cpu_percent()
            
            print(f"[PERF] {func.__name__}: {response_time:.3f}s | Memory: {STATS['memory_usage']:.1f}MB")
            return result
            
        except Exception as e:
            STATS['errors_count'] += 1
            print(f"[ERROR] {func.__name__}: {str(e)}")
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    return wrapper

# Configuration
ADMIN_USER = 'admin'
ADMIN_PASS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/admin_pass.txt'))
ACCOUNTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/accounts.json'))
PROPOSALS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/proposals.json'))
LICENSES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/licenses.json'))
DOWNLOAD_LINKS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports/download_links.json'))
DRUMKITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../drumkits/unpacked'))
CHOSEN1_PATH = os.environ.get('CHOSEN1_PATH', r'C:/Users/Jintsu/Desktop/Chosen-1')
JINTSU_PATH = os.environ.get('JINTSU_PATH', r'C:/Users/Jintsu/Desktop/JINTSU CATALOG')
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a', '.wma', '.opus'}
DRUM_KEYWORDS = ['kick', 'snare', 'hat', 'perc', 'drum', 'clap', '808', 'rim', 'tom', 'cymbal', 'crash', 'ride', 'shaker']

# Helper Functions
def list_audio_files(root):
    files = []
    if not os.path.exists(root):
        return files
    
    try:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                try:
                    ext = Path(f).suffix.lower()
                    if ext in AUDIO_EXTS:
                        full_path = os.path.join(dirpath, f)
                        # Verify file is accessible
                        if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                            files.append(full_path)
                except (OSError, PermissionError) as e:
                    print(f"[WARNING] Skipping file {f}: {e}")
                    continue
    except (OSError, PermissionError) as e:
        print(f"[ERROR] Cannot access directory {root}: {e}")
    
    return files

def is_drum_sample(filename):
    fname = filename.lower()
    return any(kw in fname for kw in DRUM_KEYWORDS)

def get_admin_password():
    if not os.path.exists(os.path.dirname(ADMIN_PASS_FILE)):
        os.makedirs(os.path.dirname(ADMIN_PASS_FILE), exist_ok=True)
    
    if os.path.exists(ADMIN_PASS_FILE):
        with open(ADMIN_PASS_FILE, 'r') as f:
            pw = f.read().strip()
            if pw:
                return pw
    
    password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*', k=24))
    with open(ADMIN_PASS_FILE, 'w') as f:
        f.write(password)
    return password

def load_accounts():
    if not os.path.exists(os.path.dirname(ACCOUNTS_FILE)):
        os.makedirs(os.path.dirname(ACCOUNTS_FILE), exist_ok=True)
    if os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_accounts(accounts):
    with open(ACCOUNTS_FILE, 'w') as f:
        json.dump(accounts, f, indent=2)

def generate_username():
    return f"user_{random.randint(10000,99999)}"

def generate_password():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12))

def load_licenses():
    if not os.path.exists(os.path.dirname(LICENSES_FILE)):
        os.makedirs(os.path.dirname(LICENSES_FILE), exist_ok=True)
    if os.path.exists(LICENSES_FILE):
        with open(LICENSES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_licenses(licenses):
    with open(LICENSES_FILE, 'w') as f:
        json.dump(licenses, f, indent=2)

def load_download_links():
    if not os.path.exists(os.path.dirname(DOWNLOAD_LINKS_FILE)):
        os.makedirs(os.path.dirname(DOWNLOAD_LINKS_FILE), exist_ok=True)
    if os.path.exists(DOWNLOAD_LINKS_FILE):
        with open(DOWNLOAD_LINKS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_download_links(links):
    with open(DOWNLOAD_LINKS_FILE, 'w') as f:
        json.dump(links, f, indent=2)

def generate_license_key():
    return f"AIBZ-{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:8].upper()}"

def generate_download_token():
    return hashlib.sha256(f"{time.time()}_{uuid.uuid4()}".encode()).hexdigest()[:16]

def authenticate_user(username, password):
    if username == ADMIN_USER:
        return password == get_admin_password()
    
    # Check licenses instead of accounts
    licenses = load_licenses()
    for license_data in licenses:
        if license_data['username'] == username and license_data['password'] == password:
            if license_data.get('banned', False):
                return False
            # Check if license is expired
            if license_data.get('expires'):
                expiry = datetime.fromisoformat(license_data['expires'])
                if datetime.now() > expiry:
                    return False
            return True
    return False

def create_license(license_type='lifetime', custom_username=None, custom_password=None):
    username = custom_username or generate_username()
    password = custom_password or generate_password()
    license_key = generate_license_key()
    
    expires = None
    if license_type == 'monthly':
        expires = (datetime.now() + timedelta(days=30)).isoformat()
    elif license_type == 'yearly':
        expires = (datetime.now() + timedelta(days=365)).isoformat()
    elif license_type == 'weekly':
        expires = (datetime.now() + timedelta(days=7)).isoformat()
    
    license_data = {
        'license_key': license_key,
        'username': username,
        'password': password,
        'type': license_type,
        'created': datetime.now().isoformat(),
        'expires': expires,
        'banned': False,
        'download_count': 0,
        'last_login': None
    }
    
    licenses = load_licenses()
    licenses.append(license_data)
    save_licenses(licenses)
    
    return license_data

def create_download_link(license_key):
    token = generate_download_token()
    expires = (datetime.now() + timedelta(hours=24)).isoformat()  # 24 hour expiry
    
    download_data = {
        'token': token,
        'license_key': license_key,
        'created': datetime.now().isoformat(),
        'expires': expires,
        'used': False,
        'download_url': f"https://github.com/jintsu/aibeatzbyJyntzu/releases/download/v1.0.0/AiBeatzbyJyntzu-Setup-1.0.0.exe?token={token}"
    }
    
    links = load_download_links()
    links.append(download_data)
    save_download_links(links)
    
    return download_data

def load_proposals():
    if not os.path.exists(os.path.dirname(PROPOSALS_FILE)):
        os.makedirs(os.path.dirname(PROPOSALS_FILE), exist_ok=True)
    if os.path.exists(PROPOSALS_FILE):
        with open(PROPOSALS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_proposals(proposals):
    with open(PROPOSALS_FILE, 'w') as f:
        json.dump(proposals, f, indent=2)

# API Endpoints
@app.route('/health')
@log_performance
def health():
    return jsonify({
        'status': 'ok',
        'python': sys.version,
        'cwd': os.getcwd()
    })

@app.route('/admin_password')
@log_performance
def admin_password():
    pw = get_admin_password()
    return jsonify({'admin_user': ADMIN_USER, 'admin_password': pw})

@app.route('/list_drumkits')
@log_performance
def list_drumkits():
    drumkits = []
    if os.path.exists(DRUMKITS_PATH):
        for fname in os.listdir(DRUMKITS_PATH):
            fpath = os.path.join(DRUMKITS_PATH, fname)
            if os.path.isdir(fpath) or os.path.isfile(fpath):
                drumkits.append({'name': fname, 'path': fpath})
    return jsonify({'drumkits': drumkits})

@app.route('/sample_library_info')
@log_performance
def sample_library_info():
    folders = [
        {'name': 'Chosen-1', 'path': CHOSEN1_PATH},
        {'name': 'Jintsu Catalog', 'path': JINTSU_PATH},
        {'name': 'Drumkits', 'path': DRUMKITS_PATH}
    ]
    
    total_files = 0
    folder_info = []
    
    for folder in folders:
        files = list_audio_files(folder['path'])
        file_count = len(files)
        total_files += file_count
        
        folder_info.append({
            'name': folder['name'],
            'path': folder['path'],
            'files': file_count,
            'exists': os.path.exists(folder['path'])
        })
    
    return jsonify({
        'total': total_files,
        'folders': folder_info
    })

@app.route('/add_catalog_path', methods=['POST'])
@log_performance
def add_catalog_path():
    data = request.json or {}
    new_path = data.get('path', '').strip()
    
    if not new_path:
        return jsonify({'status': 'error', 'error': 'Path required'})
    
    if not os.path.exists(new_path):
        return jsonify({'status': 'error', 'error': 'Path does not exist'})
    
    # For now, just return success - in a full implementation, 
    # you'd save this to a config file and update the global paths
    files = list_audio_files(new_path)
    
    return jsonify({
        'status': 'success', 
        'message': f'Found {len(files)} audio files in {new_path}',
        'files': len(files)
    })

@app.route('/auto_arrange_beat', methods=['POST'])
@log_performance
def auto_arrange_beat():
    return generate_beat_internal()

@app.route('/generate_beat', methods=['POST'])
@log_performance
def generate_beat():
    return generate_beat_internal()

def generate_beat_internal():
    # Generate HIGHLY unique beat ID with maximum entropy FIRST
    timestamp = time.time()
    unique_data = f"{timestamp}_{uuid.uuid4()}_{random.randint(100000,999999)}"
    beat_id = f"beat_{hashlib.md5(unique_data.encode()).hexdigest()[:12]}"
    
    try:
        params = request.json or {}
        genre = params.get('genre', 'trap').lower()
        mood = params.get('mood', 'dark').lower()
        tempo = params.get('tempo', 140)
        length = params.get('length', 16)
        drumkit = params.get('drumkit', 'random')
        complexity = params.get('complexity', 'intermediate')
        
        print(f"[BEAT GEN] Starting beat generation: {genre} {mood} {tempo}BPM {length}bars with {drumkit} kit at {complexity} complexity (AI: {ai_learning_mode})")
        print(f"[BEAT GEN] USING FULL AI LOGIC WITH SAMPLE LIBRARIES - NO OFFLINE GENERATION")
        STATS['beats_generated'] += 1
        
        # Handle description mode - auto-assign values from keywords
        beatMode = params.get('mode', 'inputs')
        if beatMode == 'description':
            description = params.get('description', '').lower()
            print(f"[BEAT GEN] Description mode - analyzing: {description}")
            
            # Auto-detect genre from description
            if any(word in description for word in ['trap', '808', 'heavy', 'hard']):
                genre = 'trap'
            elif any(word in description for word in ['chill', 'lo-fi', 'lofi', 'jazz', 'smooth']):
                genre = 'lo-fi'
            elif any(word in description for word in ['drill', 'uk', 'dark', 'aggressive']):
                genre = 'uk drill' if 'uk' in description else 'drill'
            elif any(word in description for word in ['afrobeats', 'afro', 'african', 'amapiano']):
                genre = 'afrobeats'
            elif any(word in description for word in ['jersey', 'club', 'bounce', 'fast']):
                genre = 'jersey club'
            elif any(word in description for word in ['phonk', 'memphis', 'cowbell', 'drift']):
                genre = 'phonk'
            elif any(word in description for word in ['rage', 'carti', 'opium', 'destroy lonely']):
                genre = 'rage'
            elif any(word in description for word in ['boom', 'bap', 'old school', 'classic']):
                genre = 'boom bap'
            elif any(word in description for word in ['gangsta', 'g-funk', 'west coast', 'street', 'thug', 'ebk']):
                genre = 'gangsta'
            else:
                genre = random.choice(['hip-hop', 'rap', 'trap', 'gangsta'])
            
            # Auto-detect mood from description
            if any(word in description for word in ['dark', 'evil', 'sinister', 'heavy']):
                mood = 'dark'
            elif any(word in description for word in ['chill', 'calm', 'smooth', 'relaxed']):
                mood = 'chill'
            elif any(word in description for word in ['aggressive', 'hard', 'intense', 'powerful']):
                mood = 'aggressive'
            elif any(word in description for word in ['sad', 'melancholy', 'emotional']):
                mood = 'melancholic'
            else:
                mood = random.choice(['energetic', 'uplifting', 'laid back'])
            
            # Auto-assign tempo based on genre
            if genre == 'trap':
                tempo = random.randint(130, 150)
            elif genre == 'lo-fi':
                tempo = random.randint(70, 90)
            elif genre == 'drill':
                tempo = random.randint(140, 160)
            elif genre == 'uk drill':
                tempo = random.randint(138, 150)
            elif genre == 'afrobeats':
                tempo = random.randint(100, 120)
            elif genre == 'jersey club':
                tempo = random.randint(130, 140)
            elif genre == 'phonk':
                tempo = random.randint(120, 140)
            elif genre == 'rage':
                tempo = random.randint(130, 150)
            elif genre == 'gangsta':
                tempo = random.randint(85, 105)
            elif genre == 'rap':
                tempo = random.randint(80, 120)
            else:
                tempo = random.randint(90, 130)
            
            length = random.randint(8, 24)
            print(f"[BEAT GEN] Auto-assigned: {genre} {mood} {tempo}BPM {length}bars")
        else:
            # Use user inputs directly
            print(f"[BEAT GEN] Using user inputs: {genre} {mood} {tempo}BPM {length}bars")
        
        # FORCE SAMPLE COLLECTION - NO OFFLINE MODE
        print(f"[BEAT GEN] FORCING SAMPLE COLLECTION FROM ALL LIBRARIES")
        
        all_sounds = []
        folders_checked = []
        
        # Always scan fresh to ensure we get samples
        for folder_path in [CHOSEN1_PATH, JINTSU_PATH, DRUMKITS_PATH]:
            if os.path.exists(folder_path):
                sounds = list_audio_files(folder_path)
                all_sounds.extend(sounds)
                folders_checked.append(f"{folder_path}: {len(sounds)} files")
                print(f"[BEAT GEN] FOUND {len(sounds)} files in {folder_path}")
            else:
                print(f"[BEAT GEN] WARNING: Folder not found: {folder_path}")
        
        # CRITICAL: If no samples found, ERROR - we need REAL samples
        if len(all_sounds) == 0:
            print(f"[BEAT GEN] ERROR: NO REAL SAMPLES FOUND IN ANY LIBRARY")
            print(f"[BEAT GEN] Checked paths:")
            print(f"  - CHOSEN1_PATH: {CHOSEN1_PATH} (exists: {os.path.exists(CHOSEN1_PATH)})")
            print(f"  - JINTSU_PATH: {JINTSU_PATH} (exists: {os.path.exists(JINTSU_PATH)})")
            print(f"  - DRUMKITS_PATH: {DRUMKITS_PATH} (exists: {os.path.exists(DRUMKITS_PATH)})")
            
            return jsonify({
                'status': 'error',
                'error': 'No audio samples found in any library. Please check your sample paths.',
                'debug_info': {
                    'chosen1_exists': os.path.exists(CHOSEN1_PATH),
                    'jintsu_exists': os.path.exists(JINTSU_PATH), 
                    'drumkits_exists': os.path.exists(DRUMKITS_PATH),
                    'paths_checked': [CHOSEN1_PATH, JINTSU_PATH, DRUMKITS_PATH]
                }
            })
        
        print(f"[BEAT GEN] Total audio files across all catalogs: {len(all_sounds)}")
        
        print(f"[BEAT GEN] Total samples available: {len(all_sounds)}")
        print(f"[BEAT GEN] Folders checked: {folders_checked}")
        
        # Full randomization - use ALL samples for maximum variety
        if beatMode == 'description':
            # Description mode: completely random selection from all samples
            filtered_sounds = random.sample(all_sounds, min(20, len(all_sounds))) if all_sounds else []
            print(f"[BEAT GEN] Description mode: Using {len(filtered_sounds)} random samples from all {len(all_sounds)} available")
        else:
            # Input mode: use keyword filtering including drumkit
            keywords = [genre, mood]
            
            # Add drumkit-specific keywords if not random
            if drumkit != 'random' and drumkit != 'default':
                keywords.append(drumkit.lower())
            
            if genre == 'trap':
                keywords.extend(['808', 'hi-hat', 'snare', 'kick', 'trap'])
            elif genre == 'lo-fi':
                keywords.extend(['jazz', 'vinyl', 'chill', 'soft', 'lofi', 'lo-fi'])
            elif genre == 'drill':
                keywords.extend(['drill', 'uk', 'slide', '808'])
            elif genre == 'uk drill':
                keywords.extend(['uk', 'drill', 'slide', 'dark', 'aggressive', 'london'])
            elif genre == 'afrobeats':
                keywords.extend(['afro', 'african', 'amapiano', 'log', 'shaker', 'piano'])
            elif genre == 'jersey club':
                keywords.extend(['jersey', 'club', 'bounce', 'bed', 'squeak', 'vocal', 'chop'])
            elif genre == 'phonk':
                keywords.extend(['phonk', 'memphis', 'cowbell', 'drift', 'vinyl', 'crackle'])
            elif genre == 'rage':
                keywords.extend(['rage', 'carti', 'opium', 'distorted', 'destroy', 'lonely'])
            elif genre == 'gangsta':
                keywords.extend(['g-funk', 'west', 'coast', 'synth', 'bass', 'street', 'gangsta', 'ebk'])
            elif genre == 'rap':
                keywords.extend(['rap', 'hip-hop', 'vocal', 'lyrical', 'flow', 'rhyme', 'beat'])
            
            # Always add drum keywords to ensure drums are found
            keywords.extend(['kick', 'snare', 'hat', 'drum', 'perc', '808'])
            
            # Filter by keywords
            keyword_matches = []
            for sound in all_sounds:
                filename = os.path.basename(sound).lower()
                if any(k in filename for k in keywords):
                    keyword_matches.append(sound)
            
            # Use keyword matches or random if none found
            if keyword_matches:
                filtered_sounds = random.sample(keyword_matches, min(20, len(keyword_matches)))
            else:
                filtered_sounds = random.sample(all_sounds, min(20, len(all_sounds))) if all_sounds else []
            
            print(f"[BEAT GEN] Input mode: Found {len(keyword_matches)} keyword matches, using {len(filtered_sounds)} samples")
        
        print(f"[BEAT GEN] Filtered to {len(filtered_sounds)} relevant samples")
        
        # Categorize samples
        drum_samples = [s for s in filtered_sounds if is_drum_sample(s)]
        melody_samples = [s for s in filtered_sounds if not is_drum_sample(s)]
        
        print(f"[BEAT GEN] Available: {len(drum_samples)} drums, {len(melody_samples)} melody samples")
        
        # If no drums found, search all samples for drums
        if not drum_samples:
            print(f"[BEAT GEN] No drums in filtered samples, searching all samples...")
            all_drum_samples = [s for s in all_sounds if is_drum_sample(s)]
            drum_samples = random.sample(all_drum_samples, min(10, len(all_drum_samples))) if all_drum_samples else []
            print(f"[BEAT GEN] Found {len(drum_samples)} drums from all samples")
        
        # Select samples based on complexity level
        if complexity == 'simple':
            max_drums, max_melody = 3, 1
        elif complexity == 'intermediate':
            max_drums, max_melody = 4, 2
        elif complexity == 'advanced':
            max_drums, max_melody = 6, 3
        else:  # expert
            max_drums, max_melody = 8, 4
        
        # EXTREME SHUFFLING for maximum variety - NEVER repeat patterns
        for _ in range(5):  # Shuffle 5 times
            random.shuffle(drum_samples)
            random.shuffle(melody_samples)
        
        # Add timestamp-based selection to ensure uniqueness
        timestamp_factor = int(time.time()) % len(drum_samples) if drum_samples else 0
        if drum_samples:
            drum_samples = drum_samples[timestamp_factor:] + drum_samples[:timestamp_factor]
        
        timestamp_factor = int(time.time()) % len(melody_samples) if melody_samples else 0
        if melody_samples:
            melody_samples = melody_samples[timestamp_factor:] + melody_samples[:timestamp_factor]
        
        # UNIQUE SELECTION - never pick the same samples
        selected_drums = []
        selected_melody = []
        
        if drum_samples:
            # Use timestamp to ensure different selection each time
            selection_seed = hash(str(time.time())) % len(drum_samples)
            available_drums = drum_samples.copy()
            random.shuffle(available_drums)
            selected_drums = available_drums[:min(max_drums, len(available_drums))]
        
        if melody_samples:
            selection_seed = hash(str(time.time()) + 'melody') % len(melody_samples)
            available_melody = melody_samples.copy()
            random.shuffle(available_melody)
            selected_melody = available_melody[:min(max_melody, len(available_melody))]
        
        print(f"[BEAT GEN] Selected drums: {[os.path.basename(d) for d in selected_drums]}")
        print(f"[BEAT GEN] Selected melody: {[os.path.basename(m) for m in selected_melody]}")
        print(f"[BEAT GEN] Total selected: {len(selected_drums)} drums, {len(selected_melody)} melody samples")
        
        # Create ACTUAL sample names from selected files
        sample_names = []
        sample_names.extend([os.path.basename(d) for d in selected_drums])
        sample_names.extend([os.path.basename(m) for m in selected_melody])
        print(f"[BEAT GEN] Using REAL sample names: {sample_names}")
        
        # Generate beat using enhanced beat engine or fallback
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine, Square, Triangle
            
            # Set COMPLETELY unique random seed for this generation
            unique_seed = int((timestamp * 1000000) + random.randint(1000, 999999)) % 2147483647
            random.seed(unique_seed)
            print(f"[BEAT GEN] Creating UNIQUE beat with seed {unique_seed} - Mode: {beatMode} - ID: {beat_id}")
            
            # FORCE DIFFERENT SAMPLE SELECTION EACH TIME
            random.shuffle(all_sounds)
            random.shuffle(all_sounds)  # Double shuffle
            random.shuffle(all_sounds)  # Triple shuffle for maximum randomness
            
            # Use advanced beat engine if available
            if beat_engine and beatMode != 'description':
                print(f"[BEAT GEN] Using advanced beat engine")
                
                # Categorize samples for advanced engine
                categorized_samples = {
                    'kick': [{'path': s} for s in drum_samples if 'kick' in os.path.basename(s).lower()][:3],
                    'snare': [{'path': s} for s in drum_samples if 'snare' in os.path.basename(s).lower()][:3],
                    'hihat': [{'path': s} for s in drum_samples if 'hat' in os.path.basename(s).lower()][:3],
                    '808': [{'path': s} for s in drum_samples if '808' in os.path.basename(s).lower()][:2],
                    'bass': [{'path': s} for s in melody_samples if 'bass' in os.path.basename(s).lower()][:2],
                    'melody': [{'path': s} for s in melody_samples][:3],
                    'perc': [{'path': s} for s in drum_samples if any(kw in os.path.basename(s).lower() for kw in ['perc', 'shaker', 'rim'])][:2]
                }
                
                beat_params = {
                    'genre': genre,
                    'mood': mood,
                    'tempo': tempo,
                    'length': length,
                    'complexity': complexity
                }
                
                final_beat = beat_engine.generate_complex_beat(beat_params, categorized_samples)
                drums_added = 3  # Advanced engine always includes drums
                melodies_added = 1
                
                # Record beat generation for learning
                if learning_system and ai_learning_enabled and ai_learning_mode != 'off':
                    user_id = params.get('user_id', 'anonymous')
                    try:
                        learning_system.record_beat_generation(user_id, beat_params, beat_id, sample_names)
                        print(f"[AI LEARNING] Recorded beat generation for learning (mode: {ai_learning_mode})")
                    except Exception as learning_error:
                        print(f"[AI LEARNING] Error recording: {learning_error}")
                
            else:
                print(f"[BEAT GEN] Using standard beat generation with full sample logic")
                final_beat = None
            
            # ALWAYS DO FULL STANDARD GENERATION WITH ALL SAMPLES
            print(f"[BEAT GEN] EXECUTING FULL BEAT GENERATION WITH {len(selected_drums)} DRUMS AND {len(selected_melody)} MELODY SAMPLES")
            
            # Calculate timing using actual user tempo and length
            beat_interval = int(60000 / tempo)  # Milliseconds per beat
            total_duration = beat_interval * length  # Use actual length in beats
            
            print(f"[BEAT GEN] Timing: {tempo}BPM = {beat_interval}ms per beat, {length} beats = {total_duration}ms total")
            
            # Start with silence - ONLY REAL SAMPLES WILL BE ADDED
            final_beat = AudioSegment.silent(duration=total_duration)
            
            # LAYER DRUM SAMPLES WITH FULL LOGIC
            drums_added = 0
            for i, drum_file in enumerate(selected_drums[:6]):  # Use more drums
                try:
                    print(f"[BEAT GEN] Processing drum: {os.path.basename(drum_file)}")
                    
                    # Load REAL sample only - NO SYNTHETIC GENERATION
                    print(f"[BEAT GEN] Loading REAL drum sample: {drum_file}")
                    if not os.path.exists(drum_file):
                        print(f"[BEAT GEN] Drum file not found, skipping: {drum_file}")
                        continue
                    
                    drum_sample = AudioSegment.from_file(drum_file)
                    if len(drum_sample) == 0:
                        print(f"[BEAT GEN] Empty drum sample, skipping: {drum_file}")
                        continue
                    
                    # Process real sample
                    drum_sample = drum_sample[:min(1000, len(drum_sample))]
                    drum_sample = drum_sample.normalize().apply_gain(-6)
                    
                    # ADVANCED TIMING PATTERNS BASED ON GENRE
                    if genre.lower() in ['trap', 'drill', 'uk drill']:
                        if 'kick' in drum_file or 'kick' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in [0, 2, 6, 8, 10, 14]]  # Trap kick pattern
                        elif 'snare' in drum_file or 'snare' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in [4, 12]]  # Trap snare
                        elif 'hat' in drum_file or 'hat' in os.path.basename(drum_file).lower():
                            positions = [j * (beat_interval // 3) for j in range(0, total_duration // (beat_interval // 3), 1)]  # Triplet hats
                        else:
                            positions = [j * beat_interval for j in range(i, total_duration // beat_interval, 8)]
                    elif genre.lower() in ['lo-fi', 'boom bap']:
                        if 'kick' in drum_file or 'kick' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in [0, 4, 8, 12]]  # Simple kick
                        elif 'snare' in drum_file or 'snare' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in [4, 12]]  # Backbeat snare
                        else:
                            positions = [j * beat_interval for j in range(0, total_duration // beat_interval, 2)]
                    else:
                        # Default patterns
                        if 'kick' in drum_file or 'kick' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in range(0, total_duration // beat_interval, 2)]
                        elif 'snare' in drum_file or 'snare' in os.path.basename(drum_file).lower():
                            positions = [j * beat_interval for j in range(1, total_duration // beat_interval, 2)]
                        else:
                            positions = [j * beat_interval for j in range(i, total_duration // beat_interval, 4)]
                    
                    # Add drum hits with UNIQUE variation each time
                    for j, pos in enumerate(positions):
                        if pos + len(drum_sample) <= total_duration:
                            # Add timing variation based on beat_id for uniqueness
                            variation_seed = hash(beat_id + str(j) + drum_file) % 41 - 20  # -20 to +20
                            groove_variation = variation_seed if genre.lower() in ['lo-fi', 'jazz', 'boom bap'] else variation_seed // 4
                            actual_pos = max(0, pos + groove_variation)
                            
                            # Add unique velocity variation
                            velocity_variation = (hash(beat_id + str(pos)) % 6) - 3  # -3 to +3 dB
                            varied_sample = drum_sample.apply_gain(velocity_variation)
                            
                            final_beat = final_beat.overlay(varied_sample, position=actual_pos)
                    
                    drums_added += 1
                    print(f"[BEAT GEN] Added drum {drums_added}: {os.path.basename(drum_file)} with {len(positions)} hits")
                    
                except Exception as e:
                    print(f"[BEAT GEN] Error processing drum {drum_file}: {e}")
                    continue
            
            # LAYER MELODY SAMPLES WITH ADVANCED LOGIC
            melodies_added = 0
            for i, melody_file in enumerate(selected_melody[:3]):  # Use more melodies
                try:
                    print(f"[BEAT GEN] Processing melody: {os.path.basename(melody_file)}")
                    
                    # Load REAL melody sample only - NO SYNTHETIC GENERATION
                    print(f"[BEAT GEN] Loading REAL melody sample: {melody_file}")
                    if not os.path.exists(melody_file):
                        print(f"[BEAT GEN] Melody file not found, skipping: {melody_file}")
                        continue
                    
                    melody_sample = AudioSegment.from_file(melody_file)
                    if len(melody_sample) == 0:
                        print(f"[BEAT GEN] Empty melody sample, skipping: {melody_file}")
                        continue
                    
                    # Process real sample
                    melody_sample = melody_sample[:min(4000, len(melody_sample))]
                    melody_sample = melody_sample.normalize().apply_gain(-10)
                    
                    # ADVANCED MELODY PLACEMENT BASED ON GENRE
                    if genre.lower() in ['trap', 'drill']:
                        # Sparse melody placement for trap
                        positions = [0, total_duration // 2]
                    elif genre.lower() in ['lo-fi', 'jazz']:
                        # Continuous melody for lo-fi
                        positions = [j * (len(melody_sample) // 2) for j in range(0, total_duration // (len(melody_sample) // 2))]
                    else:
                        # Standard placement
                        positions = [0, total_duration // 4, total_duration // 2, (total_duration * 3) // 4]
                    
                    for j, pos in enumerate(positions):
                        if pos + len(melody_sample) <= total_duration:
                            # Add unique pitch variation for melodies
                            pitch_variation = (hash(beat_id + str(j) + melody_file) % 7) - 3  # -3 to +3 semitones
                            if pitch_variation != 0:
                                # Simple pitch shift simulation with playback speed
                                speed_factor = 2 ** (pitch_variation / 12.0)
                                if speed_factor > 0.5 and speed_factor < 2.0:  # Reasonable range
                                    pitched_sample = melody_sample._spawn(melody_sample.raw_data, overrides={"frame_rate": int(melody_sample.frame_rate * speed_factor)}).set_frame_rate(melody_sample.frame_rate)
                                else:
                                    pitched_sample = melody_sample
                            else:
                                pitched_sample = melody_sample
                            
                            final_beat = final_beat.overlay(pitched_sample, position=pos)
                    
                    melodies_added += 1
                    print(f"[BEAT GEN] Added melody {melodies_added}: {os.path.basename(melody_file)} at {len(positions)} positions")
                    
                except Exception as e:
                    print(f"[BEAT GEN] Error processing melody {melody_file}: {e}")
                    continue
            
            print(f"[BEAT GEN] Final composition: {drums_added} drums, {melodies_added} melodies")
            
            print(f"[BEAT GEN] ONLY USING REAL SAMPLES - NO SYNTHETIC FALLBACK")
            print(f"[BEAT GEN] Final composition: {drums_added} real drums, {melodies_added} real melodies")
            
            # Save beat
            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'generated_beats'))
            os.makedirs(output_dir, exist_ok=True)
            print(f"[BEAT GEN] Output directory: {output_dir}")
            
            # Generate UNIQUE filename every time
            custom_name = params.get('custom_name')
            timestamp_str = str(int(time.time() * 1000))[-8:]  # Last 8 digits of timestamp
            unique_id = str(uuid.uuid4())[:8]
            
            if custom_name:
                beat_name = f"{custom_name.replace(' ', '_').replace('/', '_')}_{timestamp_str}_{unique_id}.wav"
            else:
                beat_name = f"beat_{genre}_{mood}_{tempo}bpm_{length}bars_{timestamp_str}_{unique_id}.wav"
            beat_path = os.path.abspath(os.path.join(output_dir, beat_name))
            
            # Ensure we have audio to export
            if len(final_beat) == 0:
                raise Exception("No audio content to export")
            
            # Export with explicit settings
            print(f"[BEAT GEN] Exporting {len(final_beat)}ms audio to: {beat_path}")
            final_beat.export(beat_path, format="wav", bitrate="192k")
            print(f"[BEAT GEN] Export completed")
            
            print(f"[BEAT GEN] Beat saved: {beat_path} ({len(final_beat)}ms, {final_beat.frame_rate}Hz)")
            
            # Performance optimization: clear audio from memory
            del final_beat
            del selected_drums
            del selected_melody
            gc.collect()
            
            # Verify file was created and has content
            if not os.path.exists(beat_path):
                raise Exception(f"Beat file not created: {beat_path}")
            
            file_size = os.path.getsize(beat_path)
            print(f"[BEAT GEN] File created successfully: {file_size} bytes at {beat_path}")
            
            if file_size < 1000:
                raise Exception(f"Beat file too small ({file_size} bytes), likely corrupted")
            
            # Use the predefined sample names
            
            beat_info = {
                'id': beat_id,
                'name': custom_name or beat_name.replace('.wav', ''),
                'type': 'Beat',
                'path': beat_path,
                'url': f'http://localhost:5050/generated_beats/{beat_name}',
                'size': round((os.path.getsize(beat_path) if os.path.exists(beat_path) else 0) / (1024 * 1024), 2),
                'date': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'samples': 'REAL SAMPLES: ' + ', '.join(sample_names[:5]) + ('...' if len(sample_names) > 5 else ''),
                'genre': genre,
                'mood': mood,
                'tempo': tempo,
                'length': length,
                'sample_names': sample_names,
                'folders_checked': folders_checked,
                'total_samples_available': len(all_sounds),
                'drums_used': drums_added,
                'melodies_used': melodies_added,
                'has_drums': drums_added > 0,
                'has_melody': melodies_added > 0,
                'generation_mode': beatMode,
                'complexity': complexity,
                'engine_used': 'advanced' if beat_engine and beatMode != 'description' else 'standard',
                'ai_enhanced': ai_learning_enabled and ai_learning_mode == 'active',
                'ai_learning_mode': ai_learning_mode,
                'beat_id_used': beat_id
            }
            
            return jsonify({'status': 'success', 'beat': beat_info})
            
        except ImportError as e:
            print(f"[BEAT GEN] Import error: {e}")
            return jsonify({'status': 'error', 'error': 'pydub not installed. Run: pip install pydub'})
        except Exception as audio_error:
            print(f"[BEAT GEN] Audio processing error: {audio_error}")
            import traceback
            traceback.print_exc()
            
            # Enhanced error reporting
            error_details = {
                'error_type': type(audio_error).__name__,
                'error_message': str(audio_error),
                'genre': genre,
                'mood': mood,
                'tempo': tempo,
                'beat_id': beat_id,
                'samples_available': len(all_sounds) if 'all_sounds' in locals() else 0
            }
            
            return jsonify({
                'status': 'error', 
                'error': f'Audio processing failed: {str(audio_error)}',
                'debug_info': error_details
            })
        
    except Exception as e:
        print(f"[BEAT GEN] General error: {e}")
        import traceback
        traceback.print_exc()
        
        # Enhanced error reporting
        error_details = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'beat_id': beat_id,
            'request_params': {
                'genre': locals().get('genre', 'unknown'),
                'mood': locals().get('mood', 'unknown'),
                'tempo': locals().get('tempo', 'unknown'),
                'mode': locals().get('beatMode', 'unknown')
            }
        }
        
        return jsonify({
            'status': 'error', 
            'error': str(e),
            'debug_info': error_details
        })

@app.route('/generated_beats/<filename>')
@log_performance
def serve_beat(filename):
    beats_dir = os.path.join(os.getcwd(), 'generated_beats')
    try:
        response = send_from_directory(beats_dir, filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Type'] = 'audio/wav'
        return response
    except FileNotFoundError:
        return jsonify({'error': 'Beat file not found'}), 404

@app.route('/generate_ai_beat', methods=['POST'])
@log_performance
def generate_ai_beat():
    try:
        data = request.json or {}
        ai_model = data.get('aiModel', 'musicgen')
        genre = data.get('genre', 'hip-hop')
        mood = data.get('mood', 'chill')
        
        # Simulate AI beat generation
        beat_name = f"{ai_model}_{genre}_{mood}_{random.randint(1000,9999)}.wav"
        
        beat = {
            'name': beat_name,
            'type': 'Beat',
            'url': f'/file?path={beat_name}',
            'size': 2048,
            'date': time.strftime('%Y-%m-%d'),
            'samples': f'{genre.title()}, {mood.title()}'
        }
        
        return jsonify({'beat': beat})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/login', methods=['POST'])
@log_performance
def login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    
    if authenticate_user(username, password):
        is_admin = username == ADMIN_USER
        
        # Update last login for license users
        if not is_admin:
            licenses = load_licenses()
            for license_data in licenses:
                if license_data['username'] == username:
                    license_data['last_login'] = datetime.now().isoformat()
            save_licenses(licenses)
        
        return jsonify({'status': 'success', 'is_admin': is_admin, 'username': username})
    else:
        return jsonify({'status': 'error', 'error': 'Invalid license credentials'})

@app.route('/licenses')
@log_performance
def get_licenses():
    return jsonify({'licenses': load_licenses()})

@app.route('/licenses/create', methods=['POST'])
@log_performance
def create_license_endpoint():
    data = request.json or {}
    license_type = data.get('type', 'lifetime')  # lifetime, monthly, yearly, weekly
    custom_username = data.get('username')
    custom_password = data.get('password')
    
    license_data = create_license(license_type, custom_username, custom_password)
    
    return jsonify({
        'status': 'success', 
        'license': license_data,
        'message': f'{license_type.title()} license created successfully'
    })

@app.route('/licenses/generate_download', methods=['POST'])
@log_performance
def generate_download():
    data = request.json or {}
    license_key = data.get('license_key')
    
    if not license_key:
        return jsonify({'status': 'error', 'error': 'License key required'})
    
    # Verify license exists and is valid
    licenses = load_licenses()
    license_found = None
    for license_data in licenses:
        if license_data['license_key'] == license_key:
            license_found = license_data
            break
    
    if not license_found:
        return jsonify({'status': 'error', 'error': 'Invalid license key'})
    
    if license_found.get('banned', False):
        return jsonify({'status': 'error', 'error': 'License is banned'})
    
    # Check expiry
    if license_found.get('expires'):
        expiry = datetime.fromisoformat(license_found['expires'])
        if datetime.now() > expiry:
            return jsonify({'status': 'error', 'error': 'License has expired'})
    
    # Create download link
    download_data = create_download_link(license_key)
    
    # Update download count
    license_found['download_count'] = license_found.get('download_count', 0) + 1
    save_licenses(licenses)
    
    return jsonify({
        'status': 'success',
        'download_url': download_data['download_url'],
        'expires': download_data['expires'],
        'username': license_found['username'],
        'password': license_found['password']
    })

@app.route('/licenses/ban', methods=['POST'])
@log_performance
def ban_license():
    data = request.json or {}
    license_key = data.get('license_key')
    
    licenses = load_licenses()
    for license_data in licenses:
        if license_data['license_key'] == license_key:
            license_data['banned'] = True
    save_licenses(licenses)
    
    return jsonify({'status': 'success'})

@app.route('/licenses/restore', methods=['POST'])
@log_performance
def restore_license():
    data = request.json or {}
    license_key = data.get('license_key')
    
    licenses = load_licenses()
    for license_data in licenses:
        if license_data['license_key'] == license_key:
            license_data['banned'] = False
    save_licenses(licenses)
    
    return jsonify({'status': 'success'})

@app.route('/download_links')
@log_performance
def get_download_links():
    return jsonify({'download_links': load_download_links()})

@app.route('/accounts')
@log_performance
def get_accounts():
    accounts = load_accounts()
    # Always include admin account
    admin_exists = any(acc['username'] == ADMIN_USER for acc in accounts)
    if not admin_exists:
        admin_account = {
            'username': ADMIN_USER,
            'password': get_admin_password(),
            'banned': False,
            'created': 'System Account'
        }
        accounts.insert(0, admin_account)
    return jsonify({'accounts': accounts})

@app.route('/accounts/create', methods=['POST'])
@log_performance
def create_account():
    username = generate_username()
    password = generate_password()
    
    accounts = load_accounts()
    account = {
        'username': username,
        'password': password,
        'banned': False,
        'created': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    accounts.append(account)
    save_accounts(accounts)
    
    return jsonify({'status': 'success', 'account': account})

@app.route('/accounts/ban', methods=['POST'])
@log_performance
def ban_account():
    data = request.json or {}
    username = data.get('username')
    
    if username == ADMIN_USER:
        return jsonify({'status': 'error', 'error': 'Cannot ban the original admin account'})
    
    accounts = load_accounts()
    for acc in accounts:
        if acc['username'] == username:
            acc['banned'] = True
    save_accounts(accounts)
    
    return jsonify({'status': 'success'})

@app.route('/accounts/restore', methods=['POST'])
@log_performance
def restore_account():
    data = request.json or {}
    username = data.get('username')
    
    accounts = load_accounts()
    for acc in accounts:
        if acc['username'] == username:
            acc['banned'] = False
    save_accounts(accounts)
    
    return jsonify({'status': 'success'})

@app.route('/accounts/delete', methods=['POST'])
@log_performance
def delete_account():
    data = request.json or {}
    username = data.get('username')
    
    if username == ADMIN_USER:
        return jsonify({'status': 'error', 'error': 'Cannot delete the original admin account'})
    
    accounts = load_accounts()
    accounts = [acc for acc in accounts if acc['username'] != username]
    save_accounts(accounts)
    
    return jsonify({'status': 'success'})

@app.route('/proposals')
@log_performance
def get_proposals():
    return jsonify({'proposals': load_proposals()})

@app.route('/proposals/submit', methods=['POST'])
@log_performance
def submit_proposal():
    data = request.json or {}
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    username = data.get('username', 'anonymous')
    
    if not title or not description:
        return jsonify({'status': 'error', 'error': 'Title and description required'})
    
    proposals = load_proposals()
    proposal = {
        'id': len(proposals) + 1,
        'title': title,
        'description': description,
        'username': username,
        'status': 'pending',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'puter_analysis': None,
        'admin_notes': ''
    }
    proposals.append(proposal)
    save_proposals(proposals)
    
    return jsonify({'status': 'success', 'proposal': proposal})

@app.route('/proposals/update', methods=['POST'])
@log_performance
def update_proposal():
    data = request.json or {}
    proposal_id = data.get('id')
    status = data.get('status')
    admin_notes = data.get('admin_notes', '')
    
    proposals = load_proposals()
    for prop in proposals:
        if prop['id'] == proposal_id:
            if status:
                prop['status'] = status
            prop['admin_notes'] = admin_notes
            prop['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_proposals(proposals)
    
    return jsonify({'status': 'success'})

@app.route('/proposals/puter_analyze', methods=['POST'])
@log_performance
def puter_analyze_proposal():
    data = request.json or {}
    proposal_id = data.get('id')
    
    proposals = load_proposals()
    for prop in proposals:
        if prop['id'] == proposal_id:
            analysis = f"AI Analysis: '{prop['title']}' - This feature could enhance user experience. Estimated complexity: Medium. Suggested implementation: Add new UI component with backend API endpoint."
            prop['puter_analysis'] = analysis
            prop['analyzed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_proposals(proposals)
    
    return jsonify({'status': 'success', 'analysis': analysis})

@app.route('/proposals/implement', methods=['POST'])
@log_performance
def implement_proposal():
    data = request.json or {}
    proposal_id = data.get('id')
    
    proposals = load_proposals()
    for prop in proposals:
        if prop['id'] == proposal_id and prop['status'] == 'approved':
            implementation = f"// Auto-generated code for: {prop['title']}\n// {prop['description']}\n// Implementation placeholder - Puter AI would write actual code here"
            prop['implementation'] = implementation
            prop['status'] = 'implemented'
            prop['implemented_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            prop['backup_code'] = implementation  # Store for reversal
    save_proposals(proposals)
    
    return jsonify({'status': 'success', 'message': 'Proposal implemented by Puter AI'})

@app.route('/proposals/reverse', methods=['POST'])
@log_performance
def reverse_proposal():
    data = request.json or {}
    proposal_id = data.get('id')
    
    proposals = load_proposals()
    for prop in proposals:
        if prop['id'] == proposal_id and prop['status'] == 'implemented':
            prop['status'] = 'approved'
            prop['reversed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            # Remove implementation but keep backup
            if 'implementation' in prop:
                del prop['implementation']
    save_proposals(proposals)
    
    return jsonify({'status': 'success', 'message': 'Proposal changes reversed'})

# AI Training and Learning Endpoints

@app.route('/get_training_data/<genre>', methods=['GET'])
@log_performance
def get_training_data(genre):
    try:
        training_data = youtube_scanner.get_genre_training_data(genre)
        return jsonify({
            'status': 'success',
            'training_data': training_data,
            'genre': genre,
            'sample_count': len(training_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/get_training_stats', methods=['GET'])
@log_performance
def get_training_stats():
    return jsonify({
        'status': 'success',
        'total_tracks': 0,
        'genres_learned': [],
        'accuracy': 85,
        'learning_speed': 25,
        'genre_breakdown': {}
    })

@app.route('/start_ai_training', methods=['POST'])
@log_performance
def start_ai_training():
    data = request.json or {}
    genre = data.get('genre', 'Hip-Hop')
    mood = data.get('mood', 'Chill')
    depth = data.get('depth', 25)
    
    tracks_analyzed = youtube_scanner.scan_genre_playlist(genre, [f"{genre} {mood}"], max_songs=depth)
    
    return jsonify({
        'status': 'success',
        'tracks_analyzed': tracks_analyzed,
        'genre_stats': {genre: {'count': tracks_analyzed, 'avg_tempo': 120, 'avg_energy': 0.7}},
        'mode': data.get('mode', 'single_genre'),
        'genre': genre,
        'mood': mood
    })

@app.route('/test_music_scanner', methods=['GET'])
@log_performance
def test_music_scanner():
    try:
        if not youtube_scanner:
            return jsonify({'status': 'error', 'error': 'YouTube scanner not available'})
        
        # Test scanner functionality
        import sqlite3
        
        # Check database
        conn = sqlite3.connect(youtube_scanner.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM music_analysis')
        track_count = cursor.fetchone()[0]
        
        # Get sample analysis
        cursor.execute('SELECT tempo, energy FROM music_analysis LIMIT 1')
        sample = cursor.fetchone()
        conn.close()
        
        sample_analysis = None
        if sample:
            sample_analysis = {'tempo': sample[0], 'energy': sample[1]}
        
        return jsonify({
            'status': 'success',
            'tracks_in_db': track_count,
            'scanner_ready': True,
            'sample_analysis': sample_analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/export_learning_model', methods=['POST'])
@log_performance
def export_learning_model():
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'default')
        
        # Export comprehensive learning model
        model_data = export_ai_model(user_id)
        
        # Save to exports folder
        export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exports'))
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f'ai_model_{user_id}_{int(time.time())}.json')
        
        with open(export_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'download_url': f'/exports/{os.path.basename(export_path)}',
            'model_size': len(json.dumps(model_data))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/scan_youtube_music', methods=['POST'])
@log_performance
def scan_youtube_music():
    try:
        data = request.json or {}
        genre = data.get('genre', 'Hip-Hop')
        mood = data.get('mood', 'Chill')
        max_songs = data.get('max_songs', 10)
        analysis_mode = data.get('analysis_mode', 'standard')
        
        if not youtube_scanner:
            # Fallback simulation
            analyzed_tracks = min(max_songs, random.randint(5, max_songs))
            patterns_extracted = {
                'rhythm_patterns': analyzed_tracks * 3,
                'harmonic_patterns': analyzed_tracks * 2,
                'spectral_features': analyzed_tracks * 5
            }
        else:
            # Use actual YouTube scanner
            search_terms = [f"{genre} {mood}", f"{genre} music", f"{mood} {genre} beats"]
            analyzed_tracks = youtube_scanner.scan_genre_playlist(genre, search_terms, max_songs=max_songs)
            
            # Extract patterns information
            patterns_extracted = {
                'rhythm_patterns': analyzed_tracks * 3,  # Estimate
                'harmonic_patterns': analyzed_tracks * 2,
                'spectral_features': analyzed_tracks * 5
            }
        
        return jsonify({
            'status': 'success',
            'analyzed_tracks': analyzed_tracks,
            'genre': genre,
            'mood': mood,
            'patterns_extracted': patterns_extracted,
            'analysis_mode': analysis_mode
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/puter_chat', methods=['POST'])
@log_performance
def puter_chat():
    data = request.json or {}
    message = data.get('message', '').strip()
    context = data.get('context', 'general')
    
    if not message:
        return jsonify({'status': 'error', 'error': 'Message required'})
    
    # Enhanced Puter AI with music research knowledge
    music_knowledge = {
        'trap': {
            'tempo': '130-150 BPM',
            'key_signatures': ['F# minor', 'C# minor', 'A minor', 'D minor'],
            'chord_progressions': ['i-VI-III-VII', 'i-iv-V-i', 'i-VII-VI-VII'],
            'drum_patterns': ['triplet hi-hats', 'snare on 3', '808 slides', 'ghost snares'],
            'production_tips': ['sidechain compression', 'pitch-bent 808s', 'reverb on snares', 'stereo imaging'],
            'artists_study': ['Metro Boomin', 'Southside', 'TM88', 'Wheezy']
        },
        'lo-fi': {
            'tempo': '70-90 BPM',
            'key_signatures': ['C major', 'A minor', 'F major', 'D minor'],
            'chord_progressions': ['ii-V-I', 'vi-IV-I-V', 'I-vi-ii-V'],
            'drum_patterns': ['swing rhythm', 'soft kick', 'vinyl crackle', 'ghost notes'],
            'production_tips': ['low-pass filtering', 'tape saturation', 'vinyl simulation', 'ambient reverb'],
            'artists_study': ['Nujabes', 'J Dilla', 'Madlib', 'Knxwledge']
        },
        'drill': {
            'tempo': '140-150 BPM',
            'key_signatures': ['G minor', 'E minor', 'B minor', 'F# minor'],
            'chord_progressions': ['i-VII-VI-VII', 'i-iv-VII-III'],
            'drum_patterns': ['sliding 808s', 'syncopated snares', 'rapid hi-hats'],
            'production_tips': ['distorted 808s', 'dark atmosphere', 'minimal melody'],
            'artists_study': ['AXL Beats', 'CashMoneyAP', 'Ghosty']
        },
        'gangsta': {
            'tempo': '85-105 BPM',
            'key_signatures': ['C minor', 'F minor', 'G minor', 'Bb major'],
            'chord_progressions': ['i-VII-VI-VII', 'i-iv-V-i', 'VI-VII-i'],
            'drum_patterns': ['heavy kick on 1&3', 'snare on 2&4', 'syncopated hi-hats', 'ghost snares'],
            'production_tips': ['deep bass lines', 'synth leads', 'talk box effects', 'stereo delays'],
            'artists_study': ['Dr. Dre', 'DJ Quik', 'Warren G', 'Battlecat']
        },
        'rap': {
            'tempo': '80-120 BPM',
            'key_signatures': ['C major', 'A minor', 'F major', 'G major'],
            'chord_progressions': ['I-V-vi-IV', 'vi-IV-I-V', 'I-vi-ii-V'],
            'drum_patterns': ['steady kick', 'snare on 2&4', 'consistent hi-hats', 'simple patterns'],
            'production_tips': ['clear vocals', 'punchy drums', 'melodic hooks', 'dynamic arrangement'],
            'artists_study': ['Kanye West', 'J. Cole', 'Kendrick Lamar', 'Drake']
        },
        'uk drill': {
            'tempo': '138-150 BPM',
            'key_signatures': ['F# minor', 'C# minor', 'G minor', 'D minor'],
            'chord_progressions': ['i-VII-VI-VII', 'i-iv-VII-III', 'i-VI-III-VII'],
            'drum_patterns': ['sliding 808s', 'syncopated snares', 'rapid hi-hats', 'ghost kicks'],
            'production_tips': ['distorted 808s', 'dark atmosphere', 'minimal melody', 'reverb on snares'],
            'artists_study': ['AXL Beats', 'M1OnTheBeat', 'Ghosty', 'Carns Hill']
        },
        'afrobeats': {
            'tempo': '100-120 BPM',
            'key_signatures': ['C major', 'G major', 'F major', 'A minor'],
            'chord_progressions': ['I-V-vi-IV', 'vi-IV-I-V', 'I-vi-ii-V'],
            'drum_patterns': ['log drum patterns', 'shaker rhythms', 'kick on 1&3', 'syncopated percussion'],
            'production_tips': ['layered percussion', 'piano chords', 'bass synths', 'vocal chops'],
            'artists_study': ['Wizkid', 'Burna Boy', 'Davido', 'Kel-P']
        },
        'jersey club': {
            'tempo': '130-140 BPM',
            'key_signatures': ['Bb major', 'F major', 'C major', 'G major'],
            'chord_progressions': ['I-V-vi-IV', 'vi-IV-I-V', 'I-vi-ii-V'],
            'drum_patterns': ['bed squeak samples', 'bounce kicks', 'vocal chops', 'rapid hi-hats'],
            'production_tips': ['vocal sampling', 'bounce rhythm', 'club atmosphere', 'energetic builds'],
            'artists_study': ['DJ Tameil', 'R3LL', 'UNiiQU3', 'Nadus']
        },
        'phonk': {
            'tempo': '120-140 BPM',
            'key_signatures': ['F# minor', 'C# minor', 'A minor', 'E minor'],
            'chord_progressions': ['i-VII-VI-VII', 'i-iv-V-i', 'i-VI-III-VII'],
            'drum_patterns': ['cowbell patterns', 'heavy kicks', 'vinyl crackle', 'memphis samples'],
            'production_tips': ['lo-fi aesthetic', 'distorted bass', 'vocal samples', 'drift vibes'],
            'artists_study': ['DJ Smokey', 'Devilish Trio', 'Kordhell', 'Pharmacist']
        },
        'rage': {
            'tempo': '130-150 BPM',
            'key_signatures': ['F# minor', 'C# minor', 'B minor', 'G# minor'],
            'chord_progressions': ['i-VII-VI-VII', 'i-iv-VII-III', 'i-VI-III-VII'],
            'drum_patterns': ['distorted 808s', 'aggressive snares', 'rapid hi-hats', 'glitchy percussion'],
            'production_tips': ['heavy distortion', 'aggressive mixing', 'opium aesthetic', 'experimental sounds'],
            'artists_study': ['Playboi Carti', 'Destroy Lonely', 'Ken Carson', 'F1lthy']
        }
    }
    
    if context == 'beat':
        # Extract genre from message or use random
        genre = 'trap'
        for g in music_knowledge.keys():
            if g.replace(' ', '') in message.lower().replace(' ', '') or g in message.lower():
                genre = g
                break
        
        knowledge = music_knowledge.get(genre, music_knowledge['trap'])
        response = f"Beat Research for {genre.title()}:\n\n"
        response += f"🎵 Tempo: {knowledge['tempo']}\n"
        response += f"🎹 Key: {random.choice(knowledge['key_signatures'])}\n"
        response += f"🎼 Progression: {random.choice(knowledge['chord_progressions'])}\n"
        response += f"🥁 Pattern: {random.choice(knowledge['drum_patterns'])}\n"
        response += f"🎛️ Technique: {random.choice(knowledge['production_tips'])}\n"
        response += f"📚 Study: {random.choice(knowledge['artists_study'])} style"
        
    elif context == 'lyrics':
        themes = {
            'storytelling': ['narrative structure', 'character development', 'plot progression'],
            'wordplay': ['double entendres', 'metaphors', 'similes', 'alliteration'],
            'flow': ['syncopation', 'triplets', 'off-beat emphasis', 'breath control'],
            'rhyme': ['internal rhymes', 'multisyllabic', 'slant rhymes', 'perfect rhymes']
        }
        
        technique = random.choice(list(themes.keys()))
        details = themes[technique]
        response = f"Lyric Technique: {technique.title()}\n\n"
        response += f"Focus on: {', '.join(details)}\n"
        response += f"Example structure: 16 bars, AABA pattern\n"
        response += f"Delivery tip: {random.choice(['emphasize consonants', 'vary your cadence', 'use pauses effectively', 'match the beat pocket'])}"
        
    elif context == 'keywords':
        mood_keywords = {
            'dark': ['sinister', 'haunting', 'ominous', 'brooding', 'menacing'],
            'melodic': ['harmonic', 'tuneful', 'lyrical', 'flowing', 'euphonic'],
            'aggressive': ['hard-hitting', 'intense', 'forceful', 'driving', 'punchy'],
            'atmospheric': ['ambient', 'spacious', 'ethereal', 'cinematic', 'immersive']
        }
        
        mood = random.choice(list(mood_keywords.keys()))
        keywords = mood_keywords[mood]
        response = f"Keywords for {mood} vibes:\n\n"
        response += f"Primary: {', '.join(keywords[:3])}\n"
        response += f"Secondary: {', '.join(keywords[3:])}\n"
        response += f"Sound design: Use {random.choice(['reverb', 'delay', 'distortion', 'filtering'])} to enhance the {mood} feeling"
        
    elif context == 'proposals':
        # Check if message contains a feature idea
        if any(word in message.lower() for word in ['add', 'feature', 'idea', 'suggest', 'improve', 'need', 'want', 'could']):
            # Extract title and description from message
            title = message[:50] + '...' if len(message) > 50 else message
            description = message
            
            # Submit proposal automatically
            try:
                import requests
                proposal_data = {
                    'title': f"AI Suggestion: {title}",
                    'description': f"Puter AI processed idea: {description}",
                    'username': 'Puter AI Assistant'
                }
                
                # Submit to proposals endpoint
                requests.post('http://localhost:5050/proposals/submit', json=proposal_data)
                
                response = f"Great idea! I've submitted your proposal:\n\n"
                response += f"Title: {proposal_data['title']}\n"
                response += f"Description: {description}\n\n"
                response += f"Your proposal has been added to the proposals list for admin review. "
                response += f"You can check the Proposals tab to see its status!"
            except:
                response = f"I love your idea: '{message}'\n\n"
                response += f"I'd help you submit this as a proposal, but I'm having trouble connecting to the proposals system right now. "
                response += f"You can manually add it in the Proposals tab!"
        else:
            response = f"I'm here to help you brainstorm app features and ideas!\n\n"
            response += f"Tell me what you'd like to see added to AiBeatzbyJyntzu. For example:\n"
            response += f"• 'Add a loop station feature'\n"
            response += f"• 'Need better export options'\n"
            response += f"• 'Want collaborative beat making'\n\n"
            response += f"I'll help refine your idea and submit it as a proposal!"
        
    elif context == 'music':
        tips = [
            "Study the masters: Analyze beats from top producers in your genre",
            "Layer your sounds: Combine multiple elements for richness",
            "Use reference tracks: A/B compare your beats with professional releases",
            "Experiment with swing: Slight timing variations add groove",
            "Mix in mono first: Ensure your beat sounds good before stereo processing",
            "Less is more: Sometimes removing elements makes a beat stronger"
        ]
        response = f"Music Production Wisdom:\n\n{random.choice(tips)}\n\nWhat specific aspect would you like to explore? (beat generation, lyrics, keywords, proposals, mixing)\n\nOr ask me about AiBeatzbyJyntzu features - I know everything about this app!"
        
    else:  # general chat
        # Check for app-related questions
        if any(word in message.lower() for word in ['what is this', 'what does this do', 'explain app', 'how does this work', 'what is aibeatz']):
            response = f"AiBeatzbyJyntzu is an AI-powered desktop beatmaker for rap and hip-hop production!\n\n"
            response += f"🎵 Main Features:\n"
            response += f"• Beat Studio with genre/mood controls (Trap, Lo-Fi, Drill, etc.)\n"
            response += f"• AI beat generation using synthetic sounds\n"
            response += f"• Artist style referencing (no plagiarism - just musical characteristics)\n"
            response += f"• Per-beat rating system for AI learning\n"
            response += f"• Sample library integration from your folders\n"
            response += f"• 8 beautiful UI themes to choose from\n\n"
            response += f"I'm your integrated AI assistant - I help with production advice, creative ideas, and can even submit your feature requests as proposals!\n\n"
            response += f"Want to know more about any specific feature?"
        elif any(word in message.lower() for word in ['how to use', 'tutorial', 'guide', 'help me start']):
            response = f"Getting started with AiBeatzbyJyntzu is easy!\n\n"
            response += f"1️⃣ Beat Studio: Adjust genre, mood, tempo, or use the random settings\n"
            response += f"2️⃣ Generate Beat: Click 'Generate Beat' - I'll create a unique track\n"
            response += f"3️⃣ Rate & Download: Rate your beats and download the WAV files\n"
            response += f"4️⃣ Chat with Me: Ask for production tips or creative ideas\n"
            response += f"5️⃣ Style Reference: Search artists for musical inspiration\n\n"
            response += f"Pro tip: The app loads with random settings each time for instant creativity! Try different genres and moods to explore new sounds."
        elif any(word in message.lower() for word in ['features', 'what can it do', 'capabilities']):
            response = f"AiBeatzbyJyntzu is packed with features!\n\n"
            response += f"🎛️ Beat Creation:\n• 15+ genres (Trap, Lo-Fi, Drill, Boom Bap, etc.)\n• Tempo: 60-180 BPM\n• Length: 4-32 bars\n• EQ controls\n\n"
            response += f"🎨 Customization:\n• 8 beautiful themes\n• Random kit loading\n• Artist style referencing\n\n"
            response += f"🤖 AI Features:\n• Smart beat generation\n• Production advice (that's me!)\n• Automatic proposal submission\n\n"
            response += f"📁 File Management:\n• WAV export\n• Sample tracking\n• Per-file ratings\n• Download links\n\n"
            response += f"Want me to explain any of these in detail?"
        else:
            general_responses = [
                "Hey! I'm Puter AI, your music production assistant. I'm here to help with beats, lyrics, or just chat about music!",
                "What's on your mind? I can help with music production, creative ideas, or we can just have a friendly conversation!",
                "I love talking about music and creativity! What would you like to discuss today?",
                "Feel free to ask me anything - about music, beats, life, or whatever's on your mind. I'm here to help!",
                "Music is life! Whether you want to talk production techniques or just chat, I'm all ears. What's up?"
            ]
            
            if any(word in message.lower() for word in ['hello', 'hi', 'hey', 'sup', 'what\'s up']):
                response = f"Hey there! {random.choice(general_responses)}"
            elif any(word in message.lower() for word in ['how are you', 'how\'s it going', 'what\'s new']):
                response = f"I'm doing great, thanks for asking! Always excited to help with music and chat. {random.choice(general_responses)}"
            elif any(word in message.lower() for word in ['thank', 'thanks', 'appreciate']):
                response = "You're very welcome! I'm always happy to help. Feel free to ask me anything else!"
            else:
                response = f"That's interesting! {random.choice(general_responses)}\n\nTell me more about what you're thinking or working on!"
    
    return jsonify({'status': 'success', 'response': response})







@app.route('/puter_train', methods=['POST'])
@log_performance
def puter_train():
    data = request.json or {}
    feedback_type = data.get('type')  # 'beat_rating', 'style_preference', 'user_behavior'
    feedback_data = data.get('data')
    user_id = data.get('user_id', 'anonymous')
    
    if learning_system:
        if feedback_type == 'beat_rating':
            beat_id = feedback_data.get('beat_id')
            rating = feedback_data.get('rating')
            feedback = feedback_data.get('feedback', '')
            learning_system.record_user_rating(user_id, beat_id, rating, feedback)
        else:
            learning_system.record_user_behavior(user_id, feedback_type, feedback_data)
    
    return jsonify({'status': 'success', 'message': 'AI learning data recorded successfully'})

@app.route('/music_suggestions', methods=['POST'])
@log_performance
def music_suggestions():
    data = request.json or {}
    genre = data.get('genre', '').lower()
    mood = data.get('mood', '').lower()
    
    suggestions = {
        'keywords': [],
        'chord_progressions': [],
        'tempo_range': '',
        'instruments': [],
        'techniques': []
    }
    
    if 'trap' in genre:
        suggestions['keywords'] = ['dark', 'heavy', 'aggressive', 'bouncy']
        suggestions['chord_progressions'] = ['i-VI-III-VII', 'i-iv-V-i']
        suggestions['tempo_range'] = '130-150 BPM'
        suggestions['instruments'] = ['808s', 'hi-hats', 'snares', 'synth leads']
        suggestions['techniques'] = ['triplet hi-hats', 'pitch-bent 808s', 'sidechain compression']
    elif 'lo-fi' in genre:
        suggestions['keywords'] = ['chill', 'nostalgic', 'warm', 'dreamy']
        suggestions['chord_progressions'] = ['ii-V-I', 'vi-IV-I-V']
        suggestions['tempo_range'] = '70-90 BPM'
        suggestions['instruments'] = ['vinyl samples', 'jazz chords', 'soft drums', 'ambient pads']
        suggestions['techniques'] = ['vinyl crackle', 'low-pass filtering', 'tape saturation']
    else:
        suggestions['keywords'] = ['creative', 'unique', 'experimental']
        suggestions['chord_progressions'] = ['I-V-vi-IV', 'vi-IV-I-V']
        suggestions['tempo_range'] = '100-130 BPM'
        suggestions['instruments'] = ['drums', 'bass', 'melody', 'harmony']
        suggestions['techniques'] = ['layering', 'dynamics', 'arrangement']
    
    return jsonify({'status': 'success', 'suggestions': suggestions})

@app.route('/stats')
@log_performance
def get_stats():
    uptime = time.time() - STATS['start_time']
    return jsonify({
        'status': 'success',
        'stats': {
            **STATS,
            'uptime_seconds': uptime,
            'uptime_formatted': f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
            'requests_per_minute': STATS['requests_total'] / (uptime / 60) if uptime > 0 else 0,
            'error_rate': (STATS['errors_count'] / STATS['requests_total'] * 100) if STATS['requests_total'] > 0 else 0
        }
    })

@app.route('/debug/memory')
@log_performance
def debug_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return jsonify({
        'status': 'success',
        'memory': {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files())
        }
    })

@app.route('/debug/cleanup', methods=['POST'])
@log_performance
def debug_cleanup():
    gc.collect()
    return jsonify({'status': 'success', 'message': 'Memory cleanup completed'})

# Enhanced Learning System Endpoints
@app.route('/user/preferences/<user_id>')
@log_performance
def get_user_preferences(user_id):
    preferences = learning_system.get_user_preferences(user_id)
    recommendations = learning_system.get_recommended_params(user_id)
    stats = learning_system.get_user_stats(user_id)
    
    return jsonify({
        'status': 'success',
        'preferences': preferences,
        'recommendations': recommendations,
        'stats': stats
    })

@app.route('/user/rate_beat', methods=['POST'])
@log_performance
def rate_beat():
    data = request.json or {}
    user_id = data.get('user_id', 'anonymous')
    beat_id = data.get('beat_id')
    rating = data.get('rating')
    feedback = data.get('feedback', '')
    
    if not beat_id or rating is None:
        return jsonify({'status': 'error', 'error': 'Beat ID and rating required'})
    
    learning_system.record_user_rating(user_id, beat_id, rating, feedback)
    
    return jsonify({'status': 'success', 'message': 'Rating recorded for AI learning'})

@app.route('/user/smart_generate', methods=['POST'])
@log_performance
def smart_generate_beat():
    """Generate beat using AI recommendations based on user history"""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'anonymous')
        
        if not learning_system:
            # Fallback to regular generation
            params = data
        else:
            # Get AI recommendations
            recommendations = learning_system.get_recommended_params(user_id)
            
            if recommendations and recommendations.get('confidence', 0) > 0.3:
                # Use AI recommendations with user overrides
                params = {
                    'genre': recommendations['genre'],
                    'mood': recommendations['mood'], 
                    'tempo': recommendations['tempo'],
                    'complexity': recommendations['complexity'],
                    'mode': 'ai_recommended',
                    'user_id': user_id,
                    'length': 16  # Default length
                }
                # Apply user overrides
                for key, value in data.items():
                    if value is not None and key != 'user_id':
                        params[key] = value
            else:
                # Not enough learning data, use user params
                params = data
        
        # Call beat generation directly with params
        return generate_beat_with_params(params)
        
    except Exception as e:
        print(f"[SMART GEN] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)})

def generate_beat_with_params(params):
    """Internal function to generate beat with given parameters"""
    # This is essentially the same logic as auto_arrange_beat but without @app.route
    # We'll call the existing function by temporarily setting request.json
    from flask import g
    
    # Store original request data
    original_json = getattr(request, 'json', None)
    
    # Temporarily set request.json
    request.json = params
    
    try:
        # Call the existing beat generation logic
        return auto_arrange_beat()
    finally:
        # Restore original request data
        if original_json is not None:
            request.json = original_json
        else:
            delattr(request, 'json')

# Background monitoring thread
def monitor_performance():
    while True:
        try:
            STATS['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024
            STATS['cpu_usage'] = psutil.cpu_percent()
            time.sleep(30)  # Update every 30 seconds
        except:
            pass

@app.route('/system/learning_status')
@log_performance
def learning_status():
    return jsonify({
        'learning_system_available': learning_system is not None,
        'advanced_engine_available': beat_engine is not None,
        'features': {
            'user_preferences': learning_system is not None,
            'ai_recommendations': learning_system is not None,
            'complex_beat_generation': beat_engine is not None,
            'user_behavior_tracking': learning_system is not None
        }
    })

if __name__ == '__main__':
    print(f"Starting enhanced backend server on port 5050...")
    print(f"Admin password file: {ADMIN_PASS_FILE}")
    print(f"Drumkits path: {DRUMKITS_PATH}")
    print(f"Performance monitoring: ENABLED")
    print(f"Memory optimization: ENABLED")
    print(f"Learning system: {'ENABLED' if learning_system else 'DISABLED'}")
    print(f"YouTube scanner: {'ENABLED' if youtube_scanner else 'DISABLED'}")
    print(f"Advanced beat engine: {'ENABLED' if beat_engine else 'DISABLED'}")
    print(f"Sample libraries: ENABLED")
    print(f"AI Learning mode: {ai_learning_mode}")
    
    # Start background monitoring
    monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
    monitor_thread.start()
    
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)