import cv2
import numpy as np
from constants import *

def get_data():



def face_extract(imgPath):

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(imgPath)

    faces = face_cascade.detectMultiScale(1.1, 4)

    (x, y, w, h) = faces[0]
    img = img[y:y+h, x:x+w]
    print(str(w) + ',' + str(h))
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return img
