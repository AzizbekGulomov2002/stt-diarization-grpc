import os
import json
import re
import wget
import shutil
import librosa
import numpy as np
import requests
import logging
from tqdm import tqdm
import soundfile as sf
import torch, torchaudio
from uuid import uuid4
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from gender_pipeline import gender_pipe
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


class DiarizationPipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.stt_api = os.getenv("STT_API", "")
        self.msdd_model = os.getenv("MSDD_MODEL", "")

        logger.info("Initializing DiarizationPipeline...")
        try:
            snapshot_download(
                repo_id=self.msdd_model,
                local_dir="diarization-mssd-model",
                local_dir_use_symlinks=False,
            )
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def format_time(self, ms):
        """Convert milliseconds to 'MM:SS.SS' format"""
        seconds = ms / 1000
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:05.2f}".replace(".", "_")

    def clean_transcription_text(self, transcription):
        special_tokens = ["<|endoftext|>", "<|transcribe|>", "<|notimestamps|>", "<|uz|>"]
        for token in special_tokens:
            transcription = transcription.replace(token, "")
        return transcription.strip()

    def merge_rttm_segments(self, rttm_path, merge_threshold=100):
        logger.info(f"Merging RTTM segments from {rttm_path}...")
        segments = []
        try:
            with open(rttm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        duration = float(parts[4])
                        end = start + duration
                        speaker = parts[7]
                        segments.append((speaker, start, end))
        except FileNotFoundError:
            logger.error(f"RTTM file not found: {rttm_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading RTTM file: {e}")
            raise

        merged = []
        if not segments:
            logger.warning("No segments found in RTTM file.")
            return merged

        prev_speaker, prev_start, prev_end = segments[0]
        for speaker, start, end in segments[1:]:
            if speaker == prev_speaker and start - prev_end <= merge_threshold:
                prev_end = end
            else:
                merged.append((prev_speaker, prev_start, prev_end))
                prev_speaker, prev_start, prev_end = speaker, start, end
        merged.append((prev_speaker, prev_start, prev_end))
        logger.info(f"Merged {len(segments)} segments into {len(merged)}.")
        return merged

    def transcribe_audio(self, file_path):
        url = "http://stt-service:8000/transcribe/"
        headers = {"accept": "application/json"}

        logger.info(f"Sending audio to transcription API: {file_path}")
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path, f, "audio/mpeg")}
                response = requests.post(url, headers=headers, files=files)
        except Exception as e:
            logger.error(f"Error sending request to transcription API: {e}")
            raise

        logger.debug(f"Transcription API response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            try:
                data = response.json()
                return data.get("transcription", "")
            except json.JSONDecodeError:
                logger.error("Invalid JSON in transcription response.")
                raise
        else:
            logger.error(f"Transcription API error {response.status_code}: {response.text}")
            raise Exception(f"API error {response.status_code}")

    # def get_text_and_gender(self, waveform, sample_rate, start, end):
    #     temp_path = f"{start}-{end}.wav"
    #     try:
    #         sf.write(temp_path, waveform, sample_rate)
    #         logger.debug(f"Temporary segment saved: {temp_path}")

    #         text = self.transcribe_audio(temp_path)
    #         text = self.clean_transcription_text(text)
    #         text = re.sub(r'^\.*\s*', '', text)

    #         gender = gender_pipe(temp_path)[0]['label']
    #         logger.info(f"Segment {start}-{end}s: '{text}' ({gender})")
    #         torch.cuda.empty_cache()
    #         return text, gender
    #     finally:
    #         if os.path.exists(temp_path):
    #             os.remove(temp_path)

    # def diar_stt_merged(self, audio_path, rttm_path):
    #     logger.info(f"Processing diarization and STT merge for {audio_path}")
    #     waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    #     merged_segments = self.merge_rttm_segments(rttm_path)

    #     result = []
    #     for speaker, start, end in tqdm(merged_segments, desc="Processing segments"):
    #         start_sample = int(start * sample_rate)
    #         end_sample = int(end * sample_rate)
    #         segment_audio = waveform[start_sample:end_sample]

    #         try:
    #             text, gender = self.get_text_and_gender(segment_audio, sample_rate, start, end)
    #         except Exception as e:
    #             logger.error(f"Error processing segment {start}-{end}: {e}")
    #             text, gender = "", "unknown"

    #         result.append({
    #             "speaker": speaker,
    #             "gender": gender,
    #             "start": start,
    #             "end": end,
    #             "text": text
    #         })
    #     return result

    def get_text_and_gender(self, waveform, sample_rate):
        # stereo -> mono
        if waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0, keepdims=True)  # shape [1, num_samples]

        output_path = f"{uuid4()}.wav"
        sf.write(output_path, waveform.squeeze(), sample_rate)

        text = self.transcribe_audio(output_path)
        text = self.clean_transcription_text(text)
        gender = gender_pipe(output_path)[0]['label']
        return text, gender

    def diar_stt_merged(self, audio_path, rttm_path,target_samplerate=16000):
        # audiolarni 16kHz ga resample qilib yuklaymiz
        waveform, sample_rate = librosa.load(audio_path, sr=target_samplerate, mono=False)  # stereo saqlab qoladi
        
        # agar stereo bo'lsa: waveform.shape = (channels, samples)
        if waveform.ndim == 1:  # mono audio
            waveform = np.expand_dims(waveform, axis=0)

        merged_segments = self.merge_rttm_segments(rttm_path)
        result = []

        for speaker, start, end in tqdm(merged_segments):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            segment_audio = waveform[:, start_sample:end_sample]  # slicing channels bo‘yicha

            text, gender = self.get_text_and_gender(segment_audio, sample_rate)

            start_str = self.format_time(int(start * 1000)).replace("_", ".")
            end_str = self.format_time(int(end * 1000)).replace("_", ".")
            result.append({
                "start": start_str,
                "end": end_str,
                "speaker": speaker,
                "gender": gender,
                "text": text
            })
        return result


    def convert_audio_rate(self, input_file, output_file):
        logger.info(f"Converting audio rate: {input_file} → {output_file}")
        audio, sr = librosa.load(input_file, sr=None, mono=True)
        sf.write(output_file, audio, sr, subtype='PCM_16')

    def process_audio(self, input_file, domain_type='telephonic'): # meeting
            data_dir = os.path.join("diarization_folder", str(uuid4()))
            os.makedirs(data_dir, exist_ok=True)
            output_file = os.path.join(data_dir, 'converted_audio.wav')
            if not os.path.exists(input_file):
                return f"File not found: {input_file}"
           
            self.convert_audio_rate(input_file, output_file)
            CONFIG_FILE_NAME = f"diar_infer_{domain_type}.yaml"
            CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
            
            if not os.path.exists(os.path.join(data_dir, CONFIG_FILE_NAME)):
                CONFIG = wget.download(CONFIG_URL, data_dir)
            else:
                CONFIG = os.path.join(data_dir, CONFIG_FILE_NAME)

            cfg = OmegaConf.load(CONFIG)
            meta = {
                'audio_filepath': output_file,
                'offset': 0,
                'duration': None,
                'label': 'infer',
                'text': '-',
                'num_speakers': 2,
                'rttm_filepath': None,
                'uem_filepath': None
            }
            with open(os.path.join(data_dir, 'input_manifest.json'), 'w') as fp:
                json.dump(meta, fp)
                fp.write('\n')
              
            cfg.name = "ClusterDiarizer"
            cfg.num_workers = 1
            cfg.sample_rate = 16000
            cfg.batch_size = 64
            cfg.device = None
            cfg.verbose = True
            cfg.diarizer.collar = 0.25
            cfg.diarizer.ignore_overlap = True
            cfg.diarizer.manifest_filepath = os.path.join(data_dir, 'input_manifest.json')
            cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
            cfg.diarizer.out_dir = data_dir

            cfg.diarizer.speaker_embeddings.model_path = "titanet_large"
            cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
            cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.25]
            cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
            cfg.diarizer.speaker_embeddings.parameters.save_embeddings = True
            
            cfg.diarizer.clustering.parameters.oracle_num_speakers = False
            cfg.diarizer.clustering.parameters.max_num_speakers = 2
            cfg.diarizer.clustering.parameters.enhanced_count_thres = 80
            cfg.diarizer.clustering.parameters.max_rp_threshold = 0.25
            cfg.diarizer.clustering.parameters.sparse_search_volume = 30
            cfg.diarizer.clustering.parameters.maj_vote_spk_count = False
            cfg.diarizer.clustering.parameters.chunk_cluster_count = 50
            cfg.diarizer.clustering.parameters.embeddings_per_chunk = 1000

            cfg.diarizer.msdd_model.model_path = "diarization-mssd-model/MultiscaleDiarDecoder.nemo"
            cfg.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = True
            cfg.diarizer.msdd_model.parameters.infer_batch_size = 25
            cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7]
            cfg.diarizer.msdd_model.parameters.seq_eval_mode = False
            cfg.diarizer.msdd_model.parameters.split_infer = True
            cfg.diarizer.msdd_model.parameters.diar_window_length = 50
            cfg.diarizer.msdd_model.parameters.overlap_infer_spk_limit = 2

            cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
            cfg.diarizer.vad.parameters.onset = 0.8
            cfg.diarizer.vad.parameters.offset = 0.6
            cfg.diarizer.vad.parameters.pad_offset = -0.05
            cfg.diarizer.oracle_vad = False


            sd_model = ClusteringDiarizer(cfg=cfg)
            sd_model.diarize()

            rttm_path=f"{data_dir}/pred_rttms/converted_audio.rttm"
            transcript = self.diar_stt_merged(input_file, rttm_path)
            shutil.rmtree(data_dir)  
            return transcript