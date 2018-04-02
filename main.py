"""Putting it all together"""

import io
import os

from google.cloud import speech, vision, language


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    "/home",
    "saubin",
    "credentials",
    "Learning Night-8b61b06f75e6.json")


def emotion_from_photo(client, file_name):
    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    assert len(faces) == 1
    face = faces[0]
    return {'sorrow': likelihood_name[face.sorrow_likelihood],
            'anger': likelihood_name[face.anger_likelihood],
            'joy': likelihood_name[face.joy_likelihood],
            'surprise': likelihood_name[face.surprise_likelihood]}


def text_from_speech(client, file_name):
    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.types.RecognitionAudio(content=content)

    # Google is very picky about the audio. Use single-channel FLAC.
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.FLAC,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    return response.results[0].alternatives[0].transcript


def emotion_from_text(client, text):
    document = language.types.Document(
        content=text,
        type=language.enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    return {"score": sentiment.score, "magnitude": sentiment.magnitude}


# Instantiate the clients
vis_client = vision.ImageAnnotatorClient()
speech_client = speech.SpeechClient()
text_client = language.LanguageServiceClient()

# Analyse the photo
img_file_name = os.path.join(
    os.path.dirname(__file__),
    'data', 'images', 'joy.jpg')
img_emo = emotion_from_photo(vis_client, img_file_name)

# Get sentiment from audio
audio_file_name = os.path.join(
    os.path.dirname(__file__),
    'data', 'speech', 'jasmine.flac')

speech_text = text_from_speech(speech_client, audio_file_name)
speech_emo = emotion_from_text(text_client, speech_text)

print(img_emo)
print(speech_emo)
