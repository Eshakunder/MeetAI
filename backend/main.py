import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import shutil
import tempfile
import traceback

# --- Basic Setup ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- API URLs ---
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
# UPDATED THE SUMMARIZATION MODEL to a more stable one
SUMMARY_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

AUTH_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main File Upload Endpoint ---
@app.post("/upload")
async def upload_and_process_file(file: UploadFile = File(...)):
    print("\n--- New Upload Request ---")
    print(f"✅ Received file: {file.filename}")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print("👍 File saved successfully.")
        except Exception:
            raise HTTPException(status_code=500, detail="Could not save uploaded file.")

        filename, extension = os.path.splitext(file_path)
        audio_path = file_path

        if extension.lower() in ['.mp4', '.mov', '.avi']:
            print("📹 Video file detected. Extracting audio...")
            try:
                video = AudioSegment.from_file(file_path, format=extension.lstrip('.'))
                audio_path = f"{filename}.mp3"
                video.export(audio_path, format="mp3")
                print("🔊 Audio extracted successfully.")
            except Exception:
                print(f"❌ ERROR during audio extraction: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="Failed to process video file.")

        # --- 1. Transcription with Whisper ---
        transcript_text = "No transcription available."
        try:
            print("🎤 Sending audio to Whisper for transcription...")
            with open(audio_path, "rb") as f:
                data = f.read()

            content_type = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
            request_headers = {
                "Authorization": AUTH_HEADERS["Authorization"],
                "Content-Type": content_type
            }

            response = requests.post(WHISPER_API_URL, headers=request_headers, data=data, timeout=300)

            if response.status_code != 200:
                print(f"❌ Whisper API Error (Status {response.status_code}): {response.text}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {response.reason}")
            
            transcript_json = response.json()
            transcript_text = transcript_json.get("text", "").strip() or "Could not transcribe audio."
            print("📝 Transcription received.")

        except Exception as e:
            print(f"❌ ERROR during transcription step: {e}")
            raise HTTPException(status_code=500, detail=f"A server error occurred during transcription: {str(e)}")


        # --- 2. Summarization ---
        summary_text = "No summary available."
        # Only try to summarize if transcription was successful
        if "Error" not in transcript_text and "Failed" not in transcript_text and transcript_text:
            try:
                print("🧠 Asking new model for a summary...")
                
                # UPDATED THE PAYLOAD: This model prefers a simpler input
                summary_payload = {"inputs": transcript_text}
                
                response = requests.post(SUMMARY_API_URL, headers=AUTH_HEADERS, json=summary_payload)

                if response.status_code == 200:
                    summary_json = response.json()
                    # UPDATED THE PARSING: This model returns a key called "summary_text"
                    summary_text = summary_json[0]['summary_text']
                    print("💡 Summary received.")
                else:
                    print(f"❌ Summary API Error (Status {response.status_code}): {response.text}")
                    summary_text = f"Error from Summary Model: {response.reason}"

            except Exception as e:
                print(f"❌ ERROR during summarization: {e}")
                summary_text = "Failed to process summary."

        print("✅ Process complete. Sending response to frontend.")
        return {"transcript": transcript_text, "summary": summary_text}