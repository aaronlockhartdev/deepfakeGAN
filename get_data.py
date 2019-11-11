import cv2
import os
import numpy as np
from constants import *
from progress.bar import IncrementalBar

def load_batches():
    files = os.listdir(DATA_OUTPUT_PATH + '/split')
    numBatches = int(len(files)/BATCH_SIZE)
    try:
        map = np.memmap(DATA_OUTPUT_PATH + "/batches.map", dtype=np.float16, mode='r+', shape=(numBatches, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    except FileNotFoundError:
        batch_data()
        map = np.memmap(DATA_OUTPUT_PATH + "/batches.map", dtype=np.float16, mode='r+', shape=(numBatches, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    return map

def load_data():
    files = os.listdir(DATA_OUTPUT_PATH + '/split')
    map = np.memmap(DATA_OUTPUT_PATH + "/data_copy.map", dtype=np.float16, mode='r+', shape=(len(files), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    return map

def map_data():
    files = os.listdir(DATA_OUTPUT_PATH + '/split')
    map = np.memmap(DATA_OUTPUT_PATH + "/data.map", dtype=np.float16, mode='w+', shape=(len(files), IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    bar = IncrementalBar('Loading', max=len(files))

    for i, f in enumerate(files):
        map[i] = np.load(DATA_OUTPUT_PATH + '/split/' + f)
        bar.next()
    del map

def batch_data():
    data = load_data()

    numBatches = int(data.shape[0]/BATCH_SIZE)

    batches = np.memmap(DATA_OUTPUT_PATH + "/batches.map", dtype=np.float16, mode='w+', shape=(numBatches, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    randomized = np.arange(numBatches * BATCH_SIZE)
    np.random.shuffle(randomized)


    bar = IncrementalBar("Batching with batch size " + str(BATCH_SIZE), max=randomized.shape[0])

    batchCount = 0
    sampleCount = 0
    for val in randomized:
        batches[batchCount][sampleCount] = data[val]
        sampleCount += 1
        if sampleCount == BATCH_SIZE:
            sampleCount = 0
            batchCount += 1
        bar.next()

    del batches

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
