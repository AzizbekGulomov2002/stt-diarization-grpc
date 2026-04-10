from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from diarization_pipeline import DiarizationPipeline
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
login(token=hf_token)
# Create FastAPI app
app = FastAPI()
# Initialize diarization pipeline
diarizer = DiarizationPipeline()

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, or .flac files are supported.")

    # Save uploaded file temporarily
    temp_id = str(uuid.uuid4())
    temp_dir = f"temp_uploads/{temp_id}"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run diarization + transcription
        result = diarizer.process_audio(file_path)

        return JSONResponse(content={"segments": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
