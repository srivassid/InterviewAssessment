import os

from pydub import AudioSegment
from pydub.silence import split_on_silence


class SplitAudioFile():

    def __init__(self):
        pass

    def split_audio_file(self, file):

        newpath_audio = os.path.join(os.getcwd(), 'audio/audio_split/')
        if not os.path.exists(newpath_audio):
            os.makedirs(newpath_audio)

        # reading from audio mp3 file
        self.sound = AudioSegment.from_wav(file)

        # spliting audio files
        self.audio_chunks = split_on_silence(self.sound, min_silence_len=500, silence_thresh=-40 )

        #loop is used to iterate over the output list
        for i, chunk in enumerate(self.audio_chunks):
           self.output_file = "audio/audio_split/chunk{0}.wav".format(i)
           print("Exporting file", self.output_file)
           chunk.export(self.output_file, format="wav")