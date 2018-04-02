import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    "C:\\", "Users", "mr_bo",
    "credentials", "Learning Night-2adf2eb95ca0.json")

# The name of the image file into memory
file_name = os.path.join(
    os.path.dirname(__file__),
    'data', 'images', 'sad_angry.jpg')
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

# Instantiate the client and send the request
client = vision.ImageAnnotatorClient()
image = types.Image(content=content)
# Detects the face and it's attributes
response = client.face_detection(image=image)
faces = response.face_annotations

# Names of likelihood from google.cloud.vision.enums
likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY',
                   'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
print('Faces:')

# Technically, more than one face can be detected
for face in faces:
    print('sorrow:', likelihood_name[face.sorrow_likelihood])
    print('anger:', likelihood_name[face.anger_likelihood])
    print('joy:', likelihood_name[face.joy_likelihood])
    print('surprise:', likelihood_name[face.surprise_likelihood])
