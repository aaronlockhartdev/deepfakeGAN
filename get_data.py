import cv2
import os
import numpy as np
from constants import *
from progress.bar import IncrementalBar

def load_data():

    return np.load(DATA_OUTPUT_PATH + 'images.npy')


def get_data():

    files = os.listdir(DATA_INPUT_PATH)

    map = np.memmap('data.map', dtype=np.float32,
                    mode='w+', shape=(200000, 3, 100, 100))

    bar = IncrementalBar('Extracting', max=len(files))

    count = 0

        def face_extract(vidPath, map):

            global count

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

                map[count] = img

                count += 1

    for i in range(len(files)):
        face_extract(DATA_INPUT_PATH + '/' + files[i], map)
        bar.next()

    np.save(DATA_OUTPUT_PATH + 'images.npy', np.asarray(dataList))


if __name__ == '__main__':
     get_data()
