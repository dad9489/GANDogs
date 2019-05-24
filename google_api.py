import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Cheer Me Up-4a9a7c926d81.json"

# Instantiates a client
client = vision.ImageAnnotatorClient()


def find_dog(filepath):

    # The name of the image file to annotate
    file_name = os.path.join(
        os.path.dirname(__file__),
        filepath)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # dog = False
    # for label in labels:
    #     if not dog:
    #         dog = (label.description == 'Dog')
    #     else:
    #         break
    if labels:
        dog = (labels[0].description == 'Dog')
        return dog
    else:
        return False
