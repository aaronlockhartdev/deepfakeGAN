import cv2
import os
import numpy as np
from constants import *
import progressbar

def load_data():

    return np.load(DATA_OUTPUT_PATH + 'images.npy')


def get_data():

    files = os.listdir(DATA_INPUT_PATH)

    dataList = []

    widgets = [
        '[', progressbar.Timer(), ']',
        progressbar.Bar(),
        '(', progressbar.AdaptiveETA(),')',
    ]

    for i in progressbar.progressbar(range(len(files))):

        face_extract(DATA_INPUT_PATH + '/' + files[i], dataList)


    np.save(DATA_OUTPUT_PATH + 'images.npy', np.asarray(dataList))



def face_extract(vidPath, dataList):

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(vidPath)

    count = 0
    success = 1

    while success:
        success, img = video.read()

        faces = faceCascade.detectMultiScale(img, 1.1, 4)

        if len(faces) == 0:
            continue

        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        dataList.append(np.copy(img))

if __name__ == '__main__':
     get_data()
