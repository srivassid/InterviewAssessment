import os

import moviepy.editor as mp

class Video_To_Audio():

    def __init__(self):
        pass

    def video_to_audio(self, file):

        newpath_chunks = os.path.join(os.getcwd(), 'audio/')

        if not os.path.exists(newpath_chunks):
            os.makedirs(newpath_chunks)

        # Insert Local Video File Path
        self.clip = mp.VideoFileClip(file)

        # Insert Local Audio File Path
        self.clip.audio.write_audiofile("audio/" + file.split('.')[0] + '.wav')

