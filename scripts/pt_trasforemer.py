import requests
import json
import os

# curl -X POST "https://bpsp9hckbdna42-8000.proxy.runpod.net/tts/extract_voice" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "audio_ref_path": "/home/nuxa/Documentos/dev/hidra/hidra-tts-models/models/es/kuv_will.mp3",
#            "output_path": "/home/nuxa/Documentos/dev/hidra/hidra-tts-models/models/es/kuv_will.pt",
#            "ref_text": "El mundo ardía en cenizas. El eco de la destrucción resonaba en una ciudad que ya no era más que un esqueleto de concreto y fuego. Desde lo alto, el paisaje era un infierno de humo y ruinas, el testamento de una batalla que había cobrado un precio demasiado alto."
#          }'
# ```

SERVER_TTS_URL = "http://localhost:8000"
SERVER_TTS_EXTRACT_ENDPOINT = "/tts/extract_voice"

def process_models(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for group in data.get("models", []):
        lang = group.get("grup_name")
        for model in group.get("models", []):
            name = model.get("name")
            audio_file = model.get("file")
            ref_text = model.get("ref_text")
            
            # Use absolute path for the original audio file
            audio_ref_path = os.path.join(base_dir, audio_file)
            
            # Creates the `pt` folder inside the `lang` folder if it doesn't exist
            output_dir = os.path.join(base_dir, "models", lang, "pt")
            os.makedirs(output_dir, exist_ok=True)
            
            # The final output pt file path 
            output_path = os.path.join(output_dir, f"{name}.pt")
            
            print(f"Processing model {name} in language {lang}...")
            
            payload = {
                "audio_ref_path": audio_ref_path,
                "output_path": output_path,
                "ref_text": ref_text
            }
            
            try:
                response = requests.post(
                    f"{SERVER_TTS_URL}{SERVER_TTS_EXTRACT_ENDPOINT}", 
                    json=payload
                )
                if response.status_code == 200:
                    print(f"Successfully extracted voice profile to {output_path}")
                else:
                    print(f"Failed to extract voice profile for {name}. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    # Get the models.json file dynamically based on the script location
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.json")
    process_models(json_path)
