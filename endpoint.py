from fastapi import FastAPI, HTTPException, Request
from datetime import datetime
from google.cloud import bigquery
import json
import logging
import requests
import whisper
import tempfile
import os
import subprocess
import magic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# BigQuery client
client = bigquery.Client()

# Load Whisper model
logger.info("Loading Whisper model...")
model = whisper.load_model("base")
logger.info("Whisper model loaded successfully")

# BigQuery table details
PROJECT_ID = "zapy-306602"
DATASET_ID = "gtms"
TABLE_ID = "events"

def inspect_file(file_path):
    """Inspect the file using ffprobe"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"File inspection result:\n{result.stdout}")
        return result.stdout
    except Exception as e:
        logger.error(f"Error inspecting file: {e}")
        return None

async def download_and_transcribe(url: str) -> str:
    temp_oga = None
    wav_path = None
    
    try:
        # Download the file with headers
        logger.info(f"Downloading audio from URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*'
        }
        
        # Stream the download to check for content
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Log response headers
        logger.info("Response headers:")
        for key, value in response.headers.items():
            logger.info(f"{key}: {value}")
        
        # Check Content-Type
        content_type = response.headers.get('Content-Type', '')
        logger.info(f"Content-Type from response: {content_type}")
        
        # Create temporary OGA file in a more accessible location
        temp_dir = os.path.join(os.path.expanduser("~"), "Downloads", "audio_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_oga_path = os.path.join(temp_dir, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.oga")
        logger.info(f"Saving audio to: {temp_oga_path}")
        
        # Download in chunks and show progress
        downloaded_size = 0
        content_length = response.headers.get('Content-Length')
        
        with open(temp_oga_path, 'wb') as temp_oga:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_oga.write(chunk)
                    downloaded_size += len(chunk)
                    if content_length:
                        progress = (downloaded_size / int(content_length)) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
            
            temp_oga.flush()
            os.fsync(temp_oga.fileno())
        
        # Verify downloaded file
        file_size = os.path.getsize(temp_oga_path)
        logger.info(f"Downloaded file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Downloaded file is empty")
        
        # Get MIME type and verify it's an audio file
        mime_type = magic.from_file(temp_oga_path, mime=True)
        logger.info(f"Detected MIME type: {mime_type}")
        
        # Read and log the first few bytes for debugging
        with open(temp_oga_path, 'rb') as f:
            header = f.read(16)
            logger.info(f"File header (hex): {header.hex()}")
        
        # Create temporary WAV file
        wav_path = os.path.join(temp_dir, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        logger.info(f"WAV file will be saved to: {wav_path}")
        
        # Try different ffmpeg approaches
        conversion_commands = [
            # Try as OGG
            ['ffmpeg', '-y', '-f', 'ogg', '-i', temp_oga_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path],
            # Try as raw audio
            ['ffmpeg', '-y', '-f', 'audio', '-i', temp_oga_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path],
            # Try without format specification
            ['ffmpeg', '-y', '-i', temp_oga_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path]
        ]
        
        success = False
        for cmd in conversion_commands:
            try:
                logger.info(f"Trying conversion with command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    success = True
                    break
                else:
                    logger.error(f"Conversion attempt failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Error during conversion attempt: {str(e)}")
        
        if not success:
            raise RuntimeError("All conversion attempts failed")
        
        # Verify the converted file
        wav_size = os.path.getsize(wav_path)
        logger.info(f"Converted WAV file size: {wav_size} bytes")
        
        if wav_size == 0:
            raise ValueError("Converted WAV file is empty")
        
        # Inspect the WAV file
        logger.info("Inspecting converted WAV file:")
        inspect_file(wav_path)
        
        # Transcribe the audio
        logger.info("Starting transcription...")
        result = model.transcribe(wav_path)
        transcription = result["text"].strip()
        
        logger.info("=" * 40)
        logger.info(f"Transcription: {transcription}")
        logger.info("=" * 40)
        
        # Keep the files for inspection
        logger.info(f"Audio files saved at:")
        logger.info(f"OGA: {temp_oga_path}")
        logger.info(f"WAV: {wav_path}")
        
        return transcription
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in download_and_transcribe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def create_event(request: Request):
    try:
        # Get raw JSON
        body = await request.json()
        logger.info(f"Received request body: {json.dumps(body, indent=2)}")
        
        # Check for audio attachments and transcribe
        transcription = None
        try:
            if (body.get("message", {}).get("attachments") and 
                len(body["message"]["attachments"]) > 0):
                
                logger.info("Found attachment in message")
                url = body["message"]["attachments"][0].get("temporary_download_url")
                if url:
                    transcription = await download_and_transcribe(url)
                    # Add transcription to the message
                    if not body["message"].get("text"):
                        body["message"]["text"] = transcription
                    body["message"]["transcription"] = transcription
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            # Continue with the request even if transcription fails
        
        # Prepare data for BigQuery
        row = {
            "created_at": datetime.utcnow().isoformat(),
            "event_name": body.get("event_type", "unknown"),
            "body": json.dumps(body)
        }
        
        # Get table reference
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        
        # Insert data into BigQuery
        errors = client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"BigQuery errors: {errors}")
            raise HTTPException(status_code=500, detail=f"Error inserting rows: {errors}")
        
        response_data = {"status": "success", "message": "Event saved successfully"}
        if transcription:
            response_data["transcription"] = transcription
            
        return response_data
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 