import os
import requests
import gradio as gr
import logging
from dotenv import load_dotenv
load_dotenv()
# 1) Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("webui")

# 2) Read env vars
STT_API = os.getenv("STT_API", "")
print("STT API",STT_API)

def transcribe_audio(file_path):
    logger.info(f"Uploading {file_path}")
    headers = {}
    with open(file_path, "rb") as f:
        response = requests.post(STT_API, files={"file": f}, headers=headers, timeout=120)
    logger.info(f"Response status: {response.status_code}")
    response.raise_for_status()
    transcript = response.json().get("segments", "")
    logger.info(f"Transcribed text: {transcript[:80]}…")
    return "\n".join([f"Line {i+1}: {line}" for i, line in enumerate(transcript)])

# 3) Build Gradio interface with basic auth
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath",label="Upload Audio"),
    outputs=gr.Textbox(label=" Transcript"),
    title="Speaker Diarization UI",
    description="Upload an audio file and receive transcription.",
    flagging_mode="never"
)

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": 7861}
    iface.launch(**launch_kwargs)
