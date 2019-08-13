# pip install SpeechRecognition
import speech_recognition as sr
from os import path
import os
import json
# brew install ffmpeg, pip install pydub
from pydub import AudioSegment
from pydub.utils import make_chunks

with open("secret_google_key.json") as f:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()

setName = 'set16'

# files
src = setName + '.mp3'
dst = setName + '.wav'

# convert wav to mp3
print('converting to wav ...')
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")

print('chunking to multiple 1 minute parts ...')
myaudio = AudioSegment.from_file(dst , "wav")
chunk_length_ms = 59000 # pydub calculates in millisec, google cloud limit = 1min
chunks = make_chunks(myaudio, chunk_length_ms)

#Export all of the individual chunks as wav files
total = []

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    chunk.export(chunk_name, format="wav")

    r = sr.Recognizer()
    data = sr.AudioFile(chunk_name)

    with data as source:
        print('calling google transcript part',str(i), 'of', len(chunks))
        # r.adjust_for_ambient_noise(source) # ambient noise removal, if needed
        audio = r.record(source)
        try:
            txt = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            total.append(txt)
        except Exception as e:
            print('made an exception')
            print(e)

print(' '.join(total))

# delete wav file
os.remove(dst)
