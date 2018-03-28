import os

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/saubin/credentials/Learning Night-8b61b06f75e6.json"

# Instantiates a client
client = language.LanguageServiceClient()

# The text to analyze
text = u'it was super awesome seeing Jasmine again'
document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment = client.analyze_sentiment(document=document).document_sentiment

print('Text: {}'.format(text))
print('Sentiment score:', sentiment.score)
print('Sentiment magnitude:', sentiment.magnitude))
