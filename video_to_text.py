import os.path

import speech_recognition as sr
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class VideoToText():

    def __init__(self):
        self.diz = {}

    def video_to_text(self,file):

        self.num_seconds_video = 60 * 5
        self.l = list(range(0, self.num_seconds_video + 1, 60))
        # print(l)

        # diz = {}
        newpath_chunks = os.path.join(os.getcwd(),'chunks/')
        if not os.path.exists(newpath_chunks):
            os.makedirs(newpath_chunks)

        newpath_converted = os.path.join(os.getcwd(), 'converted/')
        if not os.path.exists(newpath_converted):
            os.makedirs(newpath_converted)

        newpath_text = os.path.join(os.getcwd(), 'text/')
        if not os.path.exists(newpath_text):
            os.makedirs(newpath_text)

        for i in range(len(self.l) - 1):
            ffmpeg_extract_subclip(file, self.l[i] - 2 * (self.l[i] != 0), self.l[i + 1], targetname="chunks/cut{}.mp4".format(i + 1))
            self.clip = mp.VideoFileClip(r"chunks/cut{}.mp4".format(i + 1))
            self.clip.audio.write_audiofile(r"converted/converted{}.wav".format(i + 1))
            self.r = sr.Recognizer()
            self.audio = sr.AudioFile("converted/converted{}.wav".format(i + 1))
            with self.audio as source:
                self.r.adjust_for_ambient_noise(source)
                self.audio_file = self.r.record(source)
            self.result = self.r.recognize_google(self.audio_file)
            self.diz['chunk{}'.format(i + 1)] = self.result

    def write_to_file(self):
        self.l_chunks = [self.diz['chunk{}'.format(i + 1)] for i in range(len(self.diz))]
        self.text = '\n'.join(self.l_chunks)
        with open('text/recognized.txt', mode='w') as file:
            # file.write()
            # file.write("\n")
            file.write(self.text)
            print("ready!")
