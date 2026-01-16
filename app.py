from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import requests
from bs4 import BeautifulSoup
import re
import queue
import os
import glob
import numpy as np
from kokoro_onnx import Kokoro
import soundfile as sf
import io
import json
from typing import Dict, List

app = Flask(__name__)

# Bible structure (complete 66 books)
CHRISTIAN_ORDER = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
    "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
    "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke",
    "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians",
    "Galatians", "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy",
    "Titus", "Philemon", "Hebrews", "James", "1 Peter",
    "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"
]

TANAKH_ORDER = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Samuel", "Kings", "Isaiah", "Jeremiah", "Ezekiel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Psalms", "Proverbs", "Job", "Song of Solomon", "Ruth", "Lamentations",
    "Ecclesiastes", "Esther", "Daniel", "Ezra-Nehemiah", "Chronicles"
]

BOOK_CHAPTERS = {
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
    "Joshua": 24, "Judges": 21, "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24,
    "1 Kings": 22, "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36,
    "Ezra": 10, "Nehemiah": 10, "Esther": 10, "Job": 42, "Psalms": 150,
    "Proverbs": 31, "Ecclesiastes": 12, "Song of Solomon": 8, "Isaiah": 66,
    "Jeremiah": 52, "Lamentations": 5, "Ezekiel": 48, "Daniel": 12,
    "Hosea": 14, "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4,
    "Micah": 7, "Nahum": 3, "Habakkuk": 3, "Zephaniah": 3, "Haggai": 2,
    "Zechariah": 14, "Malachi": 4, "Matthew": 28, "Mark": 16, "Luke": 24,
    "John": 21, "Acts": 28, "Romans": 16, "1 Corinthians": 16, "2 Corinthians": 13,
    "Galatians": 6, "Ephesians": 6, "Philippians": 4, "Colossians": 4,
    "1 Thessalonians": 5, "2 Thessalonians": 3, "1 Timothy": 6, "2 Timothy": 4,
    "Titus": 3, "Philemon": 1, "Hebrews": 13, "James": 5, "1 Peter": 5,
    "2 Peter": 3, "1 John": 5, "2 John": 1, "3 John": 1, "Jude": 1, "Revelation": 22
}

MODELS_DIR = "kokoro_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Global streaming state
streamer_state = {
    'active': False, 'book': 'Genesis', 'chapter': 1, 'translation': 'KJV',
    'voice_model': None, 'book_order': 'christian', 'kokoro': None,
    'chapter_queue': queue.Queue(maxsize=3), 'current_chapter': 1
}

def get_chapter_text(book: str, chapter: int, translation: str = "KJV") -> str:
    """Fetch complete Bible chapter text using multiple API fallbacks."""
    
    # Try API 1: bible-api.com (supports KJV, WEB)
    try:
        url = f"https://bible-api.com/{book}+{chapter}?translation={translation.lower()}"
        print(f"Fetching from bible-api.com: {url}")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        verses = data.get('verses', [])
        
        if verses:
            full_text = ' '.join([v.get('text', '') for v in verses]).strip()
            print(f"API 1: {book} {chapter}: {len(verses)} verses, {len(full_text)} characters")
            if len(full_text) > 100:  # Sanity check
                return full_text
    except Exception as e:
        print(f"API 1 failed: {e}")
    
    # Try API 2: labs.bible.org (more complete, supports multiple translations)
    try:
        # Map translation codes
        trans_map = {'KJV': 'eng-kjv', 'WEB': 'eng-web', 'NIV': 'eng-niv', 'ESV': 'eng-esv'}
        trans_code = trans_map.get(translation.upper(), 'eng-kjv')
        
        # Format book name for API (remove spaces and numbers)
        book_clean = book.replace(' ', '').replace('1', 'I').replace('2', 'II').replace('3', 'III')
        
        url = f"https://labs.bible.org/api/?passage={book}+{chapter}&type=json"
        print(f"Fetching from labs.bible.org: {url}")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if data and isinstance(data, list):
            full_text = ' '.join([v.get('text', '') for v in data]).strip()
            print(f"API 2: {book} {chapter}: {len(data)} verses, {len(full_text)} characters")
            if len(full_text) > 100:
                return full_text
    except Exception as e:
        print(f"API 2 failed: {e}")
    
    # Try API 3: getbible.net (comprehensive, free)
    try:
        trans_map = {'KJV': 'kjv', 'WEB': 'web', 'NIV': 'niv', 'ESV': 'esv'}
        trans_code = trans_map.get(translation.upper(), 'kjv')
        
        # GetBible uses book numbers
        book_numbers = {
            'Genesis': 1, 'Exodus': 2, 'Leviticus': 3, 'Numbers': 4, 'Deuteronomy': 5,
            'Joshua': 6, 'Judges': 7, 'Ruth': 8, '1 Samuel': 9, '2 Samuel': 10,
            '1 Kings': 11, '2 Kings': 12, '1 Chronicles': 13, '2 Chronicles': 14,
            'Ezra': 15, 'Nehemiah': 16, 'Esther': 17, 'Job': 18, 'Psalms': 19,
            'Proverbs': 20, 'Ecclesiastes': 21, 'Song of Solomon': 22, 'Isaiah': 23,
            'Jeremiah': 24, 'Lamentations': 25, 'Ezekiel': 26, 'Daniel': 27,
            'Hosea': 28, 'Joel': 29, 'Amos': 30, 'Obadiah': 31, 'Jonah': 32,
            'Micah': 33, 'Nahum': 34, 'Habakkuk': 35, 'Zephaniah': 36, 'Haggai': 37,
            'Zechariah': 38, 'Malachi': 39, 'Matthew': 40, 'Mark': 41, 'Luke': 42,
            'John': 43, 'Acts': 44, 'Romans': 45, '1 Corinthians': 46, '2 Corinthians': 47,
            'Galatians': 48, 'Ephesians': 49, 'Philippians': 50, 'Colossians': 51,
            '1 Thessalonians': 52, '2 Thessalonians': 53, '1 Timothy': 54, '2 Timothy': 55,
            'Titus': 56, 'Philemon': 57, 'Hebrews': 58, 'James': 59, '1 Peter': 60,
            '2 Peter': 61, '1 John': 62, '2 John': 63, '3 John': 64, 'Jude': 65, 'Revelation': 66
        }
        
        book_num = book_numbers.get(book)
        if book_num:
            url = f"https://getbible.net/json?passage={book_num}{chapter}&version={trans_code}"
            print(f"Fetching from getbible.net: {url}")
            resp = requests.get(url, timeout=10)
            
            # GetBible returns JSONP, need to parse it
            text = resp.text
            if text.startswith('(') and text.endswith(');'):
                text = text[1:-2]
            
            data = json.loads(text)
            if 'book' in data:
                chapter_data = data['book'][0]['chapter']
                verses = []
                for verse_key in sorted(chapter_data.keys()):
                    if verse_key.isdigit():
                        verses.append(chapter_data[verse_key]['verse'])
                
                full_text = ' '.join(verses).strip()
                print(f"API 3: {book} {chapter}: {len(verses)} verses, {len(full_text)} characters")
                if len(full_text) > 100:
                    return full_text
    except Exception as e:
        print(f"API 3 failed: {e}")
    
    print(f"ERROR: All APIs failed for {book} {chapter}")
    return f"{book} {chapter} text unavailable"

def scan_models_directory():
    """Dynamic kokoro-onnx model discovery."""
    models = []
    
    # Local models
    for file in glob.glob(os.path.join(MODELS_DIR, "*.onnx")):
        name = os.path.basename(file).replace('.onnx', '')
        size = f"{os.path.getsize(file)/(1024*1024):.1f}MB"
        models.append({'name': name, 'installed': True, 'path': file, 'size': size})
    
    # Kokoro ONNX models from GitHub releases
    known_models = {
        'kokoro-v1.0': {
            'url': 'https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx',
            'voices': 'https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin'
        }
    }
    
    for name, urls in known_models.items():
        path = os.path.join(MODELS_DIR, f"{name}.onnx")
        if not os.path.exists(path):
            models.append({'name': name, 'installed': False, 'url': urls['url'], 'size': '~82MB'})
    
    return models

def download_model(name: str):
    """Download kokoro-onnx model."""
    models = scan_models_directory()
    for model in models:
        if model['name'] == name and not model.get('installed'):
            path = os.path.join(MODELS_DIR, f"{name}.onnx")
            try:
                resp = requests.get(model['url'], stream=True)
                with open(path, 'wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f"Download error: {e}")
                return False
    return False

def init_streamer(book, chapter, translation, voice_model, book_order):
    """Initialize with REAL streaming kokoro-onnx."""
    global streamer_state
    
    # Load kokoro-onnx model
    model_path = None
    voices_path = None
    for model in scan_models_directory():
        if model['name'] == voice_model and model.get('installed'):
            model_path = model['path']
            # Correct voices path: same directory, just change filename
            voices_path = os.path.join(MODELS_DIR, 'voices-v1.0.bin')
            break
    
    print(f"Model path: {model_path}")
    print(f"Voices path: {voices_path}")
    
    if model_path and os.path.exists(model_path):
        if not os.path.exists(voices_path):
            print(f"Voices file not found at {voices_path}")
            print(f"Looking for alternatives...")
            # Try alternative names
            alt_voices = [
                os.path.join(MODELS_DIR, 'voices.bin'),
                os.path.join(MODELS_DIR, f'voices-{voice_model}.bin'),
                voices_path.replace('voices-v1.0', 'voices')
            ]
            for alt in alt_voices:
                if os.path.exists(alt):
                    voices_path = alt
                    print(f"Found voices at: {voices_path}")
                    break
        
        if os.path.exists(voices_path):
            try:
                print(f"Loading Kokoro model from {model_path}...")
                streamer_state['kokoro'] = Kokoro(model_path, voices_path)
                print(f"SUCCESS: Loaded model: {model_path}")
                print(f"SUCCESS: Loaded voices: {voices_path}")
            except Exception as e:
                print(f"ERROR loading Kokoro model: {e}")
                import traceback
                traceback.print_exc()
                streamer_state['kokoro'] = None
        else:
            print(f"ERROR: Voices file missing: {voices_path}")
            print(f"Download with: wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -P {MODELS_DIR}")
            streamer_state['kokoro'] = None
    else:
        print(f"ERROR: Model not found: {voice_model}")
        streamer_state['kokoro'] = None
    
    # Setup streaming state
    streamer_state.update({
        'active': True, 'book': book, 'chapter': chapter,
        'translation': translation, 'voice_model': voice_model,
        'book_order': book_order, 'current_chapter': chapter
    })
    
    # Preload 3 chapters
    streamer_state['chapter_queue'].queue.clear()
    for i in range(3):
        chap = chapter + i
        if chap <= BOOK_CHAPTERS.get(book, 1):
            text = get_chapter_text(book, chap, translation)
            streamer_state['chapter_queue'].put((chap, text))

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def api_models():
    return jsonify({'models': scan_models_directory()})

@app.route('/api/model/<name>')
def api_model(name):
    if request.args.get('download'):
        success = download_model(name)
        return jsonify({'success': success})
    return jsonify({'error': 'No action'})

@app.route('/api/books')
def api_books():
    order = request.args.get('order', 'christian')
    books = CHRISTIAN_ORDER if order == 'christian' else TANAKH_ORDER
    return jsonify({'books': books, 'chapters': BOOK_CHAPTERS})

@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.json
    try:
        init_streamer(**data)
        return jsonify({'success': True, 'status': 'started'})
    except Exception as e:
        print(f"Error starting stream: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def api_status():
    return jsonify({
        'active': streamer_state['active'],
        'book': streamer_state['book'],
        'chapter': streamer_state['current_chapter'],
        'translation': streamer_state['translation'],
        'voice': streamer_state['voice_model'],
        'buffer': streamer_state['chapter_queue'].qsize()
    })

@app.route('/api/next_chapter')
def api_next_chapter():
    if streamer_state['chapter_queue'].empty():
        return jsonify({'end': True})
    
    chap_num, text = streamer_state['chapter_queue'].get()
    
    # Maintain buffer
    if streamer_state['chapter_queue'].qsize() < 2:
        next_chap = chap_num + 1
        if next_chap <= BOOK_CHAPTERS.get(streamer_state['book'], 1):
            next_text = get_chapter_text(streamer_state['book'], next_chap, streamer_state['translation'])
            streamer_state['chapter_queue'].put((next_chap, next_text))
    
    streamer_state['current_chapter'] = chap_num
    
    # Log text length for debugging
    print(f"Chapter {chap_num} text length: {len(text)} characters")
    
    return jsonify({
        'chapter': chap_num,
        'book': streamer_state['book'],
        'text': text[:500] + '...' if len(text) > 500 else text,
        'full_text': text,
        'text_length': len(text)
    })

@app.route('/api/stream_audio/<int:chapter>')
def api_stream_audio(chapter):
    voice = request.args.get('voice', 'af_sky')
    speed = float(request.args.get('speed', 1.0))
    sentence_index = int(request.args.get('sentence', 0))
    
    # Get book and translation from streamer_state
    book = streamer_state.get('book', 'Genesis')
    translation = streamer_state.get('translation', 'KJV')
    
    kokoro = streamer_state.get('kokoro')
    if not kokoro:
        print("ERROR: No Kokoro model loaded!")
        return jsonify({'error': 'No model loaded'}), 500
    
    try:
        text = get_chapter_text(book, chapter, translation)
        print(f"Retrieved text for {book} {chapter}: {len(text)} characters")
        
        if not text or len(text) < 10:
            print(f"ERROR: No text found for {book} {chapter}")
            return jsonify({'error': 'No text found'}), 500
        
        # Split into sentences (every 2-3 sentences = one chunk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks of 3
        chunk_size = 3
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            chunks.append(chunk)
        
        if sentence_index >= len(chunks):
            return jsonify({'error': 'End of chapter', 'total_chunks': len(chunks)}), 404
        
        chunk_text = chunks[sentence_index]
        print(f"Synthesizing {book} {chapter} chunk {sentence_index}/{len(chunks)} - {len(chunk_text)} chars")
        print(f"Voice: {voice}, Speed: {speed}x")
        
        # Synthesize this chunk
        audio_data, sample_rate = kokoro.create(chunk_text, voice=voice, speed=speed)
        print(f"Generated audio: shape={audio_data.shape}, sr={sample_rate}Hz")
        
        # Convert to WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        print(f"WAV file: {len(wav_bytes)} bytes ({len(wav_bytes)/1024:.1f} KB)")
        
        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={
                'Content-Type': 'audio/wav',
                'Content-Length': str(len(wav_bytes)),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache',
                'X-Total-Chunks': str(len(chunks)),
                'X-Current-Chunk': str(sentence_index)
            }
        )
    except Exception as e:
        print(f"ERROR: Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    streamer_state['active'] = False
    streamer_state['chapter_queue'].queue.clear()
    return jsonify({'success': True, 'status': 'stopped'})

@app.route('/api/test_audio')
def api_test_audio():
    """Test endpoint to verify audio generation works"""
    try:
        kokoro = streamer_state.get('kokoro')
        if not kokoro:
            return jsonify({'error': 'No model loaded', 'loaded': False}), 500
        
        # Generate test audio
        test_text = "This is a test of the audio system."
        audio_data, sample_rate = kokoro.create(test_text, voice='af_sky', speed=1.0)
        
        # Convert to WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        wav_bytes = buffer.read()
        
        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={
                'Content-Type': 'audio/wav',
                'Content-Length': str(len(wav_bytes))
            }
        )
    except Exception as e:
        return jsonify({'error': str(e), 'loaded': False}), 500

@app.route('/api/test_chapter')
def api_test_chapter():
    """Debug endpoint to test chapter fetching"""
    book = request.args.get('book', 'Genesis')
    chapter = int(request.args.get('chapter', 1))
    translation = request.args.get('translation', 'KJV')
    
    text = get_chapter_text(book, chapter, translation)
    
    return jsonify({
        'book': book,
        'chapter': chapter,
        'translation': translation,
        'length': len(text),
        'verse_count': text.count('\n') + 1,
        'first_100': text[:100],
        'last_100': text[-100:],
        'full_text': text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
