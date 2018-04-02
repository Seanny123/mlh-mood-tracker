import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    "C:\\", "Users", "mr_bo",
    "credentials", "Learning Night-2adf2eb95ca0.json")


# Load the audio into memory
file_name = os.path.join(
    os.path.dirname(__file__),
    'data', 'speech', 'jasmine.flac')
with io.open(file_name, 'rb') as audio_file:
    content = audio_file.read()

# Instantiate the client
client = speech.SpeechClient()
# Setup the audio request
audio = types.RecognitionAudio(content=content)
# Google is very picky about the audio. Use single-channel FLAC.
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
    language_code='en-US')
# Detect speech in the audio file
response = client.recognize(config, audio)

for result in response.results:
    print('Transcript:', result.alternatives[0].transcript)
