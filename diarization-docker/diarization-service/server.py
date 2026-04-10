import os
import tempfile
import grpc
import logging
import traceback
from concurrent import futures
from dotenv import load_dotenv
from huggingface_hub import login
from diarization_pipeline import DiarizationPipeline
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

import transcribe_pb2
import transcribe_pb2_grpc

# ------------------ Logging Setup ------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DiarizationServer")

# ------------------ Load env and HF login ------------------ #
load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    login(token=hf_token)
    logger.info("Logged into Hugging Face Hub")
else:
    logger.warning("HF_TOKEN not found in environment")

# ------------------ Initialize diarization pipeline ------------------ #
diarizer = DiarizationPipeline()

# ------------------ Audio download helper ------------------ #
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def download_audio(url: str) -> bytes:
    logger.info(f"Downloading audio from: {url}")
    response = requests.get(url, timeout=1000)
    response.raise_for_status()
    logger.info(f"Downloaded {len(response.content)} bytes from {url}")
    return response.content

def download_and_convert_to_wav(url: str, target_path: str):
    logger.info(f"Starting audio download and conversion to WAV")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".temp") as tmp_file:
        try:
            audio_data = download_audio(url)
            tmp_file.write(audio_data)
            tmp_file.flush()

            logger.info(f"Converting to WAV: {target_path}")
            audio = AudioSegment.from_file(tmp_file.name)
            audio.export(target_path, format="wav")
            logger.info(f"Conversion complete, duration: {audio.duration_seconds:.2f} sec")
            return audio.duration_seconds

        except Exception as e:
            logger.error(f"Error in download_and_convert_to_wav: {e}")
            logger.error(traceback.format_exc())
            return None
        finally:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

# ------------------ gRPC Service ------------------ #
class TranscribeServiceServicer(transcribe_pb2_grpc.TranscribeServiceServicer):
    def TranscribeAudio(self, request, context):
        audio_url = request.audio_url
        logger.info(f"Received TranscribeAudio request for URL: {audio_url}")

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "input.wav")

            # Download and convert
            duration = download_and_convert_to_wav(audio_url, wav_path)
            if duration is None:
                logger.error("Failed to download or convert audio")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to download or convert audio")
                return transcribe_pb2.TranscribeResponse()

            # Run diarization
            try:
                logger.info(f"Starting diarization for file: {wav_path}")
                diarization_results = diarizer.process_audio(wav_path)
                logger.info(f"Diarization complete, segments: {len(diarization_results)}")
            except Exception as e:
                logger.error(f"Diarization error: {e}")
                logger.error(traceback.format_exc())
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return transcribe_pb2.TranscribeResponse()

            # Build response
            speakers_set = set()
            entries = []
            logger.info("Building gRPC response from diarization results")

            for i, seg in enumerate(diarization_results):
                try:
                    text = seg.get("text", "").strip()
                    logger.debug(f"Segment {i}: {seg}")

                    if not text or text == ".":
                        logger.info(f"Skipping empty/invalid segment {i}")
                        continue

                    speakers_set.add(seg["speaker"])
                    entries.append(
                        transcribe_pb2.DiarizationEntry(
                            id=i,
                            speaker=seg["speaker"],
                            start=float(seg["start"]),
                            end=float(seg["end"]),
                            text=text,
                            gender=seg.get("gender", "")
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing segment {i}: {e}")
                    logger.error(traceback.format_exc())
                    continue  # Skip bad segment

            logger.info(f"Returning {len(entries)} valid diarization entries")
            return transcribe_pb2.TranscribeResponse(
                diarization=entries,
                audio_url=audio_url,
                speakers=list(speakers_set)
            )

# ------------------ Server Entry ------------------ #
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    transcribe_pb2_grpc.add_TranscribeServiceServicer_to_server(
        TranscribeServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    logger.info("gRPC server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
