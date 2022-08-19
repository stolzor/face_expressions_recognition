import pickle
import cv2
from cv2.dnn import readNetFromONNX
from necessary_files.preproc_sample import *


if __name__ == '__main__':
    opencv_net = readNetFromONNX('necessary_files/resnext.onnx')

    face_cascade = cv2.CascadeClassifier('necessary_files/haarcascade_frontalface_default.xml')
    label_decoder = pickle.load(open('preproc_data/label_encoder.pkl', 'rb'))

    path = input('enter path image: ')

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.putText(img, output(img[y:y + h, x:x + w], opencv_net, label_decoder), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('res_' + path, img)
    cv2.imshow('RES', img)

    cv2.waitKey(0)