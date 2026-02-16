import json
import os
import whisper
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjust base path to point to repo root assuming script runs from scripts/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_JSON_PATH = os.path.join(BASE_DIR, 'models.json')
MODEL_NAME = "medium"

def load_models_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_models_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

def transcribe_audio(file_path, whisper_model, language=None):
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
            
        logging.info(f"Transcribing {file_path} (Language: {language})...")
        # specific args for whisper if language is provided
        options = {}
        if language:
            options['language'] = language
            
        result = whisper_model.transcribe(file_path, **options)
        text = result['text'].strip()
        return text
    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {e}")
        return None

def process_single_model(model_entry, whisper_model):
    """
    Process a single model entry. Returns (bool, text) where bool is True if updated.
    """
    model_name = model_entry.get('name', 'Unknown')
    
    # Check if ref_text exists and is not empty
    if model_entry.get('ref_text'):
        logging.info(f"Skipping {model_name} - ref_text exists.")
        return False, None

    audio_file_rel = model_entry.get('file')
    if not audio_file_rel:
        logging.warning(f"Model {model_name} has no file path.")
        return False, None

    # Construct absolute path
    audio_file_abs = os.path.join(BASE_DIR, audio_file_rel)
    
    # Check if file exists, if not try replacing 'models/' with 'modeles/' (or vice versa) as fallback
    if not os.path.exists(audio_file_abs):
        if 'models/' in audio_file_rel:
            alt_rel = audio_file_rel.replace('models/', 'modeles/')
            alt_abs = os.path.join(BASE_DIR, alt_rel)
            if os.path.exists(alt_abs):
                logging.info(f"Using alternative path: {alt_rel}")
                audio_file_abs = alt_abs
        elif 'modeles/' in audio_file_rel:
             alt_rel = audio_file_rel.replace('modeles/', 'models/')
             alt_abs = os.path.join(BASE_DIR, alt_rel)
             if os.path.exists(alt_abs):
                logging.info(f"Using alternative path: {alt_rel}")
                audio_file_abs = alt_abs
    
    # Get language if available
    language = model_entry.get('language')

    # Transcribe
    text = transcribe_audio(audio_file_abs, whisper_model, language=language)
    
    if text:
        logging.info(f"Transcribed {model_name}: {text[:50]}...")
        return True, text
    
    return False, None

def main():
    if not os.path.exists(MODELS_JSON_PATH):
        logging.error(f"models.json not found at {MODELS_JSON_PATH}")
        return

    logging.info(f"Loading models from {MODELS_JSON_PATH}")
    data = load_models_json(MODELS_JSON_PATH)
    
    # Load Whisper model once
    try:
        # User requested reliability, so we force CPU to avoid CUDA VRAM issues
        logging.info(f"Loading Whisper model ({MODEL_NAME})")
        whisper_model = whisper.load_model(MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        return

    tasks = []
    
    # Traverse the structure
    root_key = 'models'
    if 'modeles' in data:
        root_key = 'modeles'
    
    if root_key in data and isinstance(data[root_key], list):
        for group in data[root_key]:
            models_key = 'models'
            if 'modeles' in group:
                models_key = 'modeles'
                
            if models_key in group and isinstance(group[models_key], list):
                for model in group[models_key]:
                    # We only care about models that need processing
                    if not model.get('ref_text'):
                        tasks.append(model)

    total_tasks = len(tasks)
    logging.info(f"Found {total_tasks} models needing reference text.")

    if total_tasks == 0:
        logging.info("No models to process.")
        return

    # Process sequentially (one by one) as requested
    updated_count = 0
    for i, model in enumerate(tasks):
        try:
            logging.info(f"Processing model {i+1}/{total_tasks}: {model.get('name')}")
            updated, text = process_single_model(model, whisper_model)
            if updated and text:
                model['ref_text'] = text
                updated_count += 1
                
                # Save immediately after successful update as requested
                logging.info(f"Saving changes for {model.get('name')}...")
                save_models_json(MODELS_JSON_PATH, data)
                
                # Small delay to ensure system stability/fiabilidad 
                # (optional, but requested 'lento')
                time.sleep(0.5) 
        except KeyboardInterrupt:
            logging.info("Process interrupted by user. Saving progress...")
            save_models_json(MODELS_JSON_PATH, data)
            return
        except Exception as exc:
            logging.error(f"Error processing {model.get('name')}: {exc}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
