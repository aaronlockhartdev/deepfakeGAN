import cv2
import os
import numpy as np
from constants import *
from progress.bar import IncrementalBar

def load_data():
    files = os.listdir(DATA_OUTPUT_PATH + '/split')
    map = np.memmap("data.map", dtype=np.float32, mode='w+', shape=(len(files), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    bar = IncrementalBar('Loading', max=len(files))

    for i, f in enumerate(files):
        map[i] = np.load(DATA_OUTPUT_PATH + '/split/' + f)
        bar.next()
    return map


def mask_data():

    files = os.listdir(DATA_INPUT_PATH)

    bar = IncrementalBar('Masking', max=len(files))

    counter = 0

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i in range(len(files)):
        video = cv2.VideoCapture(DATA_INPUT_PATH + '/' + files[i])
        success = 1
        while success:
            success, img = video.read()

            faces = faceCascade.detectMultiScale(img, 1.1, 4)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
            img = img[y:y+h, x:x+w]

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

            np.save(DATA_OUTPUT_PATH + '/split/' + str(counter) + '.npy', img)
            counter += 1

        bar.next()

if __name__ == '__main__':
     mask_data()
