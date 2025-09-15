from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
from playsound3 import playsound
class TTS:
    def __init__(self):
        self.pipeline = KPipeline(lang_code="a")
        self.voice = "af_heart"
        print("TTS loaded...")
    def text_to_speech(self, text):
        generator = self.pipeline(text, self.voice)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)
            sf.write(f'{i}.wav', audio, 24000)
            playsound(f'{i}.wav', block=True)
            break
