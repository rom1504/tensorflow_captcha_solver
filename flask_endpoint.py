from flask import Flask
import requests
from flask import request
from io import BytesIO

from captcha_solver import load_captcha_model, solve_captcha

app = Flask(__name__)

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "annotated"


def init():
    global model
    model = load_captcha_model(MODEL_FILENAME, MODEL_LABELS_FILENAME)


@app.route("/")
def solve():
    url = request.args.get('url')

    response = requests.get(url, verify=False)

    image = BytesIO(response.content)

    captcha_text = solve_captcha(image, model)

    if captcha_text == "":
        return "failure"

    return captcha_text

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)

