import cv2
import os
import numpy as np
from constants import *

def load_data():

    return np.load(DATA_OUTPUT_PATH + 'images.npy')


def get_data():

    files = os.listdir(DATA_INPUT_PATH)

    dataList = []

    for f in files:
        face_extract(DATA_INPUT_PATH + '/' + f)

    np.save(DATA_OUTPUT_PATH + 'images.npy', np.asarray(dataList))



def face_extract(vidPath):

    global list

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.videoCapture(vidPath)

    count = 0
    success = 1

    while success:
        success, img = video.read()

        faces = face_cascade.detectMultiScale(1.1, 4)

        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]

        img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        dataList.append(np.copy(img))
