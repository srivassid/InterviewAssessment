import os
import pandas as pd
from video_to_audio import Video_To_Audio
from video_emotion_detection import VideoEmotionDetection
from audio_downsample import DownsampleAudio
from split_audio_file import SplitAudioFile
from audio_emotion import AudioEmotion
from video_to_text import VideoToText
from intent_classification import IntentClassification

class InterviewAssessment():

    def __init__(self, file):
        self.file = file

    def convert_video_to_audio(self):
        self.vid_to_aud = Video_To_Audio()
        self.vid_to_aud.video_to_audio(self.file)

    def extract_emotions_from_video(self):
        self.vid_emotion_detection = VideoEmotionDetection()
        self.vid_emotion_detection.extract_emotions(self.file)

    def downsaple_audio(self):
        self.down_audio = DownsampleAudio()
        self.down_audio.downsample_audio("audio/" + self.file.split(".")[0] +\
                                         ".wav")

    def split_aud_file(self):
        self.split_aud_into_chunks = SplitAudioFile()
        self.split_aud_into_chunks.split_audio_file("audio/" + self.file.split(".")[0] +\
                                         "_down.wav")

    def aud_emotion(self):
        self.df = pd.DataFrame()
        self.sampling_rate = 16000
        self.get_audio_emotion = AudioEmotion()
        self.cwd = os.getcwd()
        self.files = os.listdir(self.cwd + "/audio/audio_split/")
        # print(self.files)
        for i in self.files:
            self.output = self.get_audio_emotion.predict("audio/audio_split/" + i, self.sampling_rate)
            self.df = self.df.append(pd.DataFrame(self.output))
            # print(self.output)
        print(self.df.shape)
        # print(self.df)
        self.df['Score'] = self.df['Score'].apply(lambda x:x.split('%')[0])
        self.df["Score"] = self.df['Score'].astype(float)
        # print(self.df)
        self.df = self.df.groupby(['Emotion']).agg('mean').reset_index()
        print(self.df.sort_values('Score',ascending=False).to_csv('audio/result.csv',index=False, header=True))

    def vid_to_text(self):
        self.vid_to_text_for_intent = VideoToText()
        self.vid_to_text_for_intent.video_to_text(self.file)
        self.vid_to_text_for_intent.write_to_file()

    def intent_detection(self):
        self.intent = IntentClassification()

        with open('text/recognized.txt') as f:
            self.text = f.readlines()

        self.recognized_intent = self.intent.get_intent(self.text)
        print(self.recognized_intent)
        with open("text/intent.txt", mode='w') as f:
            f.write(self.recognized_intent),

    def print_results(self):
        self.df = pd.read_csv('video/video_emotion_detection.csv')
        # self.df = self.df.T
        # print(self.df)
        print("Video Emotional Output")
        print("Average percentage of each emotion in each frame")
        print("Angry",self.df['angry'].mean())
        print("Disgust",self.df['disgust'].mean())
        print("Fear",self.df['fear'].mean())
        print("Happy",self.df['happy'].mean())
        print("Sad",self.df['sad'].mean())
        print("Surprise",self.df['surprise'].mean())
        print("Neutral",self.df['neutral'].mean())

        print("\n\n")

        print("Audio Emotional Output")
        self.audio_df = pd.read_csv('audio/result.csv')
        print("Average percentage of each emotion")
        print(self.audio_df)

        print("\n\n")

        print('Intent behind the text')
        with open('text/intent.txt') as f:
            print(f.readlines()[0].split('>')[1].split("<")[0])

if __name__ == '__main__':
    file = 'psy.mp4'
    interview = InterviewAssessment(file)
    interview.convert_video_to_audio()
    interview.extract_emotions_from_video()
    interview.downsaple_audio()
    interview.split_aud_file()
    interview.aud_emotion()
    interview.vid_to_text()
    interview.intent_detection()
    interview.print_results()