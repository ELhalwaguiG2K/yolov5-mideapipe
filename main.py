import os
import base64
import io
from flask import Flask, request, jsonify
from detect_modified import run
import re
import PIL.Image as Image
import numpy as np

UPLOAD_FOLDER = '\\uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/multipersonposedetection", methods=["POST"])
def upload_file():
    if request.method == 'POST':
        base64_str = request.form["imagebase64"]
        base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
        byte_data = base64.b64decode(base64_data)
        image_data = io.BytesIO(byte_data)
        img = Image.open(image_data)
        img.save("E:/Testing/yolov5+mediapipe/uploads/test.jpg")
        detectedDictionary = run()
        return jsonify({'landmarks': detectedDictionary})


if __name__ == "__main__":
    app.debug = True
    app.run()
