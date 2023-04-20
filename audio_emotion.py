import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import librosa
# import IPython.display as ipd
import numpy as np
import pandas as pd
from pydub import AudioSegment
import wave
from src.models import Wav2Vec2ForSpeechClassification

# from models import Wav2Vec2ForSpeechClassification
class AudioEmotion():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(self.model_name_or_path).to(self.device)

    def speech_file_to_array_fn(self, path, sampling_rate):
        self.speech_array, self._sampling_rate = torchaudio.load(path)
        self.resampler = torchaudio.transforms.Resample(self._sampling_rate)
        self.speech = self.resampler(self.speech_array).squeeze().numpy()
        return self.speech

    # def speech_file_to_array_fn(path, sampling_rate):
    #     speech_array, _sampling_rate = torchaudio.load(path,format="mp3")
    #     resampler = torchaudio.transforms.Resample(_sampling_rate)
    #     speech = resampler(speech_array[1]).squeeze().numpy()
    #     return speech

    def predict(self, path, sampling_rate):
        self.speech = self.speech_file_to_array_fn(path, sampling_rate)
        self.inputs = self.feature_extractor(self.speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: self.inputs[key].to(self.device) for key in self.inputs}
        with torch.no_grad():
            self.logits = self.model(**inputs).logits
        self.scores = F.softmax(self.logits, dim=1).detach().cpu().numpy()[0]
        self.outputs = [{"Emotion": self.config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(self.scores)]
        return self.outputs


