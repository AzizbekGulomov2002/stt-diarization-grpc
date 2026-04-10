from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import tempfile
import os
from transformers import pipeline
#import whisper
from huggingface_hub import login
from faster_whisper import WhisperModel
from dotenv import load_dotenv
load_dotenv()

token=os.getenv("HF_TOKEN","")
stt_model_name=os.getenv("STT_MODEL","")
login(token)
app = FastAPI(title="Whisper STT Demo")

# Load model on startup (use CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(stt_model_name, device=device, compute_type="int8_float16")
# model=pipeline("automatic-speech-recognition",
#                             model=stt_model_name,
#                             device=device,
#                             chunk_length_s=10,  
#                             generate_kwargs={
#                                 "language": "uzbek",                 # Explicitly set the language
#                                 "task": "transcribe",             # Avoid translation (which increases hallucination risk)
#                                 "num_beams": 1,                   # Disable beam search (more deterministic, fewer hallucinations)
#                                 "temperature": 0.0,               # Prevent randomness in output
#                                 "no_repeat_ngram_size": 3,        # Discourage repetitive hallucinations
#                                             },
                            
#     )
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-4:]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            segments, info = model.transcribe(
            tmp_path,
            chunk_length=30,           # activates the fast “chunked” long-form algorithm[4][6]
            beam_size=2,               # beam-5 → 2 roughly halves decoder compute with <0.2 WER drop[2]
            temperature=0.8,           # deterministic decoding avoids extra sampling passes
            vad_filter=True,           # voice-activity detection skips long silences, saving work
            condition_on_previous_text=False,  # disables costly cross-chunk conditioning when accuracy allows
            word_timestamps=False,     # turn on only if you really need them
            language="uz",
            
        )
        text = ""
        for segment in segments:
            text += segment.text + " "


        print(text)
        # Delete temp file
        os.remove(tmp_path)

        return JSONResponse(content={"transcription": text})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Whisper STT Demo is running. Use /transcribe/ to POST audio."}