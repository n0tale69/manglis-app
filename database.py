import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manglish.db")

def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Text_Corpus Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Text_Corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source_url TEXT,
            ground_truth TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Prediction_Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Prediction_Logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            cleaned_text TEXT,
            predicted_label TEXT,
            confidence_score REAL,
            processing_time_ms REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Slang_Lexicon Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Slang_Lexicon (
            slang TEXT PRIMARY KEY,
            standard_word TEXT NOT NULL,
            is_toxic INTEGER DEFAULT 0
        )
    ''')

    try:
        cursor.execute("ALTER TABLE Slang_Lexicon ADD COLUMN is_toxic INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists

    # Migrate initial toxic words to the new column
    toxic_roots = ["bodoh", "babi", "sial", "celaka", "cibai", "puki", "hinaan", "anjing", "lahanat", "gampang", "gila", "cacat", "sundal", "pelacur", "jalang", "bapok", "pondan", "gemuk", "hodoh", "buruk", "hitam", "rasis", "mati", "bunuh", "pundek", "lucah", "bodo"]
    for word in toxic_roots:
        cursor.execute("UPDATE Slang_Lexicon SET is_toxic = 1 WHERE standard_word = ?", (word,))
        cursor.execute("UPDATE Slang_Lexicon SET is_toxic = 1 WHERE slang = ?", (word,))

    conn.commit()
    conn.close()

def seed_slang_if_empty(base_dir):
    """Seed the Slang_Lexicon table from static files if it's empty."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Slang_Lexicon")
    count = cursor.fetchone()[0]

    if count == 0:
        merged = {}
        # --- source 1: malayslangdict.py ---
        slang_py_path = os.path.join(base_dir, "malayslangdict.py")
        if os.path.exists(slang_py_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("malayslangdict", slang_py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "malayslangdict"):
                merged.update(mod.malayslangdict)

        # --- source 2: manglish_dictionary.json ---
        json_path = os.path.join(base_dir, "manglish_dictionary.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                merged.update(json.load(f))

        # Insert into DB
        for slang, standard in merged.items():
            cursor.execute('''
                INSERT OR IGNORE INTO Slang_Lexicon (slang, standard_word)
                VALUES (?, ?)
            ''', (slang.lower(), standard.lower()))
        
        conn.commit()

    conn.close()

def get_slang_dict():
    """Retrieve the slang dictionary from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT slang, standard_word FROM Slang_Lexicon")
    rows = cursor.fetchall()
    conn.close()
    return {row['slang']: row['standard_word'] for row in rows}

def get_toxic_words():
    """Retrieve words flagged as toxic."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT standard_word FROM Slang_Lexicon WHERE is_toxic = 1")
        rows1 = cursor.fetchall()
        cursor.execute("SELECT DISTINCT slang FROM Slang_Lexicon WHERE is_toxic = 1")
        rows2 = cursor.fetchall()
        toxic = [row[0] for row in rows1] + [row[0] for row in rows2]
    except sqlite3.OperationalError:
        toxic = []
    conn.close()
    return set(toxic)

def log_prediction(input_text, cleaned_text, predicted_label, confidence_score, processing_time_ms):
    """Log a prediction event into Prediction_Logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Prediction_Logs (input_text, cleaned_text, predicted_label, confidence_score, processing_time_ms)
        VALUES (?, ?, ?, ?, ?)
    ''', (input_text, cleaned_text, predicted_label, confidence_score, processing_time_ms))
    conn.commit()
    conn.close()

def add_or_update_slang(slang, standard_word, is_toxic=0):
    """Add or update a slang entry."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO Slang_Lexicon (slang, standard_word, is_toxic)
        VALUES (?, ?, ?)
    ''', (slang.lower().strip(), standard_word.lower().strip(), is_toxic))
    conn.commit()
    conn.close()

def delete_slang(slang):
    """Delete a slang entry."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM Slang_Lexicon WHERE slang = ?', (slang.lower().strip(),))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=100):
    """Retrieve recent prediction logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Prediction_Logs ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    """Retrieve system statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM Prediction_Logs")
    total_analyzed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Prediction_Logs WHERE predicted_label != 'SAFE'")
    total_cyberbullying = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(confidence_score) FROM Prediction_Logs")
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(processing_time_ms) FROM Prediction_Logs")
    avg_processing_time = cursor.fetchone()[0] or 0.0

    conn.close()
    
    return {
        "total_analyzed": total_analyzed,
        "total_cyberbullying": total_cyberbullying,
        "total_safe": total_analyzed - total_cyberbullying,
        "avg_confidence": avg_confidence,
        "avg_processing_time": avg_processing_time
    }
