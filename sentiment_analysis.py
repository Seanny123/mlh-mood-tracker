import os

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    "C:\\", "Users", "mr_bo",
    "credentials", "Learning Night-2adf2eb95ca0.json")

# The text to analyze
text = 'it was super awesome seeing Jasmine again'
# Detect the sentiment of the text
client = language.LanguageServiceClient()
document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)
sentiment = client.analyze_sentiment(document=document).document_sentiment

print('Text:', text)
# Sentiment score ranges from -1 to 1
# Negative score is a negative emotion. Positive score is a positive emotion.
print('Sentiment score:', sentiment.score)
# Sentiment magnitude is how strongly the emotion is felt.
print('Sentiment magnitude:', sentiment.magnitude)
