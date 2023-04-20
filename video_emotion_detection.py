import os

import pandas as pd
from fer import Video
from fer import FER

class VideoEmotionDetection():

    def __init__(self):
        pass

    def extract_emotions(self, file):

        newpath_chunks = os.path.join(os.getcwd(), 'video/')
        if not os.path.exists(newpath_chunks):
            os.makedirs(newpath_chunks)

        video_filename = file
        self.video = Video(video_filename)

        # Analyze video, displaying the output
        self.detector = FER(mtcnn=True)
        self.raw_data = self.video.analyze(self.detector, display=True,)
        self.df = self.video.to_pandas(self.raw_data)
        print(self.df)
        self.df.to_csv('video/video_emotion_detection.csv', header=True, index=False)
        # self.df.groupby
        # self.df = pd.read_csv('video/video_emotion_detection.csv')
        # print(self.df)

