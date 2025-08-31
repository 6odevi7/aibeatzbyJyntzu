import os
import json
import sqlite3
from pathlib import Path
from pydub import AudioSegment
import librosa
import numpy as np
from datetime import datetime
import hashlib

class SampleManager:
    def __init__(self, db_path="samples.db"):
        self.db_path = db_path
        self.audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a'}
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for sample management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create samples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT,
                name TEXT,
                duration REAL,
                tempo REAL,
                key_signature TEXT,
                genre TEXT,
                mood TEXT,
                tags TEXT,
                created_date TEXT,
                last_modified TEXT,
                file_size INTEGER,
                sample_rate INTEGER,
                channels INTEGER,
                is_chopped BOOLEAN DEFAULT 0,
                parent_sample_id INTEGER,
                chop_method TEXT,
                start_time REAL,
                end_time REAL
            )
        ''')
        
        # Create collections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_date TEXT
            )
        ''')
        
        # Create sample_collections junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_collections (
                sample_id INTEGER,
                collection_id INTEGER,
                FOREIGN KEY (sample_id) REFERENCES samples (id),
                FOREIGN KEY (collection_id) REFERENCES collections (id),
                PRIMARY KEY (sample_id, collection_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def analyze_audio(self, file_path):
        """Analyze audio file and extract metadata"""
        try:
            # Load with pydub for basic info
            audio = AudioSegment.from_file(file_path)
            
            # Convert to numpy for analysis
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = samples.mean(axis=1)  # Convert to mono
            
            sample_rate = audio.frame_rate
            
            # Analyze with librosa
            tempo, _ = librosa.beat.beat_track(y=samples, sr=sample_rate)
            
            # Estimate key (simplified)
            chroma = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
            key_profile = np.mean(chroma, axis=1)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            estimated_key = key_names[np.argmax(key_profile)]
            
            return {
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'tempo': float(tempo),
                'key_signature': estimated_key,
                'sample_rate': sample_rate,
                'channels': audio.channels,
                'file_size': os.path.getsize(file_path)
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {
                'duration': 0,
                'tempo': 0,
                'key_signature': 'Unknown',
                'sample_rate': 44100,
                'channels': 1,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
    
    def add_sample(self, file_path, name=None, genre=None, mood=None, tags=None):
        """Add a sample to the database"""
        if not os.path.exists(file_path):
            return None
        
        # Calculate hash for duplicate detection
        file_hash = self.calculate_file_hash(file_path)
        
        # Check if sample already exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM samples WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return existing[0]  # Return existing ID
        
        # Analyze audio
        analysis = self.analyze_audio(file_path)
        
        # Prepare data
        if name is None:
            name = Path(file_path).stem
        
        tags_str = json.dumps(tags) if tags else None
        current_time = datetime.now().isoformat()
        
        # Insert sample
        cursor.execute('''
            INSERT INTO samples (
                file_path, file_hash, name, duration, tempo, key_signature,
                genre, mood, tags, created_date, last_modified, file_size,
                sample_rate, channels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_path, file_hash, name, analysis['duration'], analysis['tempo'],
            analysis['key_signature'], genre, mood, tags_str, current_time,
            current_time, analysis['file_size'], analysis['sample_rate'],
            analysis['channels']
        ))
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return sample_id
    
    def add_chopped_sample(self, file_path, parent_id, chop_method, start_time, end_time, name=None):
        """Add a chopped sample with reference to parent"""
        sample_id = self.add_sample(file_path, name)
        
        if sample_id:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE samples 
                SET is_chopped = 1, parent_sample_id = ?, chop_method = ?, 
                    start_time = ?, end_time = ?
                WHERE id = ?
            ''', (parent_id, chop_method, start_time, end_time, sample_id))
            
            conn.commit()
            conn.close()
        
        return sample_id
    
    def search_samples(self, query=None, genre=None, mood=None, tempo_range=None, 
                      duration_range=None, key=None, tags=None, limit=50):
        """Search samples with various filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM samples WHERE 1=1"
        params = []
        
        if query:
            sql += " AND (name LIKE ? OR file_path LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if genre:
            sql += " AND genre LIKE ?"
            params.append(f"%{genre}%")
        
        if mood:
            sql += " AND mood LIKE ?"
            params.append(f"%{mood}%")
        
        if tempo_range:
            min_tempo, max_tempo = tempo_range
            sql += " AND tempo BETWEEN ? AND ?"
            params.extend([min_tempo, max_tempo])
        
        if duration_range:
            min_duration, max_duration = duration_range
            sql += " AND duration BETWEEN ? AND ?"
            params.extend([min_duration, max_duration])
        
        if key:
            sql += " AND key_signature = ?"
            params.append(key)
        
        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        sql += f" ORDER BY created_date DESC LIMIT {limit}"
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_sample_by_id(self, sample_id):
        """Get sample by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE id = ?", (sample_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def get_chopped_samples(self, parent_id):
        """Get all chopped samples from a parent sample"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE parent_sample_id = ?", (parent_id,))
        results = cursor.fetchall()
        conn.close()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def create_collection(self, name, description=None):
        """Create a new sample collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO collections (name, description, created_date)
                VALUES (?, ?, ?)
            ''', (name, description, datetime.now().isoformat()))
            
            collection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return collection_id
        except sqlite3.IntegrityError:
            conn.close()
            return None  # Collection already exists
    
    def add_to_collection(self, sample_id, collection_id):
        """Add sample to collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO sample_collections (sample_id, collection_id)
                VALUES (?, ?)
            ''', (sample_id, collection_id))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False  # Already in collection
    
    def get_collection_samples(self, collection_id):
        """Get all samples in a collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.* FROM samples s
            JOIN sample_collections sc ON s.id = sc.sample_id
            WHERE sc.collection_id = ?
        ''', (collection_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_collections(self):
        """Get all collections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM collections ORDER BY name")
        results = cursor.fetchall()
        conn.close()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def update_sample_tags(self, sample_id, tags):
        """Update sample tags"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tags_str = json.dumps(tags) if tags else None
        cursor.execute('''
            UPDATE samples SET tags = ?, last_modified = ?
            WHERE id = ?
        ''', (tags_str, datetime.now().isoformat(), sample_id))
        
        conn.commit()
        conn.close()
    
    def delete_sample(self, sample_id):
        """Delete sample from database (not the file)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete from collections first
        cursor.execute("DELETE FROM sample_collections WHERE sample_id = ?", (sample_id,))
        
        # Delete chopped samples
        cursor.execute("DELETE FROM samples WHERE parent_sample_id = ?", (sample_id,))
        
        # Delete the sample
        cursor.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        
        conn.commit()
        conn.close()
    
    def scan_folder(self, folder_path, genre=None, mood=None, recursive=True):
        """Scan folder and add all audio files to database"""
        added_count = 0
        
        if recursive:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in self.audio_exts:
                        file_path = os.path.join(root, file)
                        
                        # Try to extract genre/mood from path
                        path_parts = Path(file_path).parts
                        detected_genre = genre
                        detected_mood = mood
                        
                        # Simple heuristics for genre/mood detection
                        for part in path_parts:
                            part_lower = part.lower()
                            if any(g in part_lower for g in ['trap', 'drill', 'boom bap', 'lo-fi', 'jazz']):
                                detected_genre = part_lower
                            if any(m in part_lower for m in ['dark', 'chill', 'aggressive', 'sad', 'happy']):
                                detected_mood = part_lower
                        
                        sample_id = self.add_sample(file_path, genre=detected_genre, mood=detected_mood)
                        if sample_id:
                            added_count += 1
                            print(f"Added: {file}")
        else:
            for file in os.listdir(folder_path):
                if Path(file).suffix.lower() in self.audio_exts:
                    file_path = os.path.join(folder_path, file)
                    sample_id = self.add_sample(file_path, genre=genre, mood=mood)
                    if sample_id:
                        added_count += 1
                        print(f"Added: {file}")
        
        return added_count
    
    def get_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total samples
        cursor.execute("SELECT COUNT(*) FROM samples")
        total_samples = cursor.fetchone()[0]
        
        # Chopped samples
        cursor.execute("SELECT COUNT(*) FROM samples WHERE is_chopped = 1")
        chopped_samples = cursor.fetchone()[0]
        
        # Total duration
        cursor.execute("SELECT SUM(duration) FROM samples")
        total_duration = cursor.fetchone()[0] or 0
        
        # Genres
        cursor.execute("SELECT genre, COUNT(*) FROM samples WHERE genre IS NOT NULL GROUP BY genre")
        genres = dict(cursor.fetchall())
        
        # Collections
        cursor.execute("SELECT COUNT(*) FROM collections")
        total_collections = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_samples': total_samples,
            'chopped_samples': chopped_samples,
            'original_samples': total_samples - chopped_samples,
            'total_duration_hours': total_duration / 3600,
            'genres': genres,
            'total_collections': total_collections
        }

# Example usage
if __name__ == "__main__":
    manager = SampleManager()
    
    # Scan drumkits folder
    drumkits_path = r"C:\Users\Jintsu\Desktop\AiBeatzbyJyntzu\drumkits\unpacked"
    if os.path.exists(drumkits_path):
        print("Scanning drumkits folder...")
        added = manager.scan_folder(drumkits_path, genre="hip-hop")
        print(f"Added {added} samples")
    
    # Print stats
    stats = manager.get_stats()
    print(f"\nDatabase Stats:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Original samples: {stats['original_samples']}")
    print(f"Chopped samples: {stats['chopped_samples']}")
    print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"Collections: {stats['total_collections']}")