from transformers import pipeline
from dotenv import load_dotenv
import os
load_dotenv()

gender_model_name = os.getenv("GENDER_MODEL", "")

gender_pipe = pipeline("audio-classification", model=gender_model_name)
