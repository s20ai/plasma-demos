import os
import cv2
import numpy as np
from PIL import Image
import logging

def getImageWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp= np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        print(ID)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, faces

def execute(base_url, path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    Ids, faces = getImageWithID(path)
    recognizer.train(faces, np.array(Ids))
    print(recognizer.write(base_url + '/models/trainingData.yml'))
    cv2.destroyAllWindows()
    return

def main(args):
    if args['component'] == 'training':
        parameters = args['parameters']
        dir = os.path.dirname(os.path.realpath(__file__)).split('/')
        dir = dir[1:len(dir)-2]
        base_url = ''
        for i in dir:
            base_url += ('/' + i)
        path = base_url + parameters['path']
        return execute(base_url, path)

