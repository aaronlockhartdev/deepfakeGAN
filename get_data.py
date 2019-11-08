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
        face_extract(DATA_INPUT_PATH + '/' + f, dataList)

    np.save(DATA_OUTPUT_PATH + 'images.npy', np.asarray(dataList))



def face_extract(vidPath, dataList):

    print("Extracting " + vidPath + "...")

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(vidPath)

    count = 0
    success = 1

    while success:
        success, img = video.read()

#	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img, 1.1, 4)

        if len(faces) == 0:
            continue
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        dataList.append(np.copy(img))

if __name__ == '__main__':
     get_data()
