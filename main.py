import os
import cv2
import uuid
import numpy as np
import imutils
from segment import Segmentator
from alignment import DocScanner
from ocrv2 import OCR

from flask import Flask, jsonify, request
import base64
from error import *
import os


app = Flask(__name__)

segmentator = Segmentator()
scanner = DocScanner()
reader = OCR()




def infer(image):
    BORDER_SIZE = int(image.shape[0] * 0.10)
    RESCALED_HEIGHT = 512

    ori_pad = cv2.copyMakeBorder(image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))
    img_pad = cv2.copyMakeBorder(image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))
    img_pad = imutils.resize(img_pad, height = int(RESCALED_HEIGHT))

    masking = segmentator.remove_background(img_pad)
    aligned_image = scanner.scan(ori_pad, masking)
    kie_info = reader.read(aligned_image)

    return kie_info

# Define your routes and their respective handlers
@app.route('/')
def hello():
    return 'Home page'

@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    try:
        image_file = request.get_json(force=True)
        image_file = image_file['image']
        decoded_data = base64.b64decode(image_file)
        image_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        
        if image.shape[0] < 512:
            #return errors['IMAGE_TOO_SMALL']
            response = {
                'date': '',
                'medicines':[],
                'diagnose':'',
                'status': 704
            }
            return jsonify(response)
        # System run
        response = infer(image)
        all_empty = all(not value for value in response.values())
        if all_empty:
            response['status'] = 702
        else:
            response['status'] = 200
        return jsonify(response)


    except:
        #jsonify(errors['NO_IMAGE_FOUND'])
        response = {
            'date': '',
            'medicines':[],
            'diagnose':'',
            'status': 703
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
