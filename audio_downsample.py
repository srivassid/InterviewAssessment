import os

import librosa
import soundfile as sf

class DownsampleAudio():

    def __init__(self):
        pass

    def downsample_audio(self, file):
        newpath_audio = os.path.join(os.getcwd(), 'audio/')
        if not os.path.exists(newpath_audio):
            os.makedirs(newpath_audio)

        self.y, self.s = librosa.load(file, sr=16000)
        sf.write( file.split('.')[0] + "_down.wav", self.y,16000)#"audio/" +

