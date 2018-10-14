from keras.models import load_model
from helpers import resize_to_fit
import numpy as np
import pickle
from image_preprocessor import preprocess_image


def load_captcha_model(model_filename = "captcha_model.hdf5"):
    # Load the trained neural network
    model = load_model(model_filename)

    return model


def load_labels(labels_filename = "model_labels.dat"):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(labels_filename, "rb") as f:
        lb = pickle.load(f)

    return lb


def solve_captcha(image_file, model, lb):
    letter_images = preprocess_image(image_file)

    if not letter_images:
        return ""

    # Create an output image and a list to hold our predicted letters
    predictions = []

    # loop over the letters
    for letter_image in letter_images :

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)

    return captcha_text
