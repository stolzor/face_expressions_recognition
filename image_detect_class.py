import pickle
import cv2
import numpy as np
from cv2.dnn import readNetFromONNX
from PIL import Image

def prepare_sample(img):
    input_img = img.astype(np.float32)
    input_img = cv2.resize(input_img, (48, 48))

    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(48, 48),
        mean=mean
    )
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    input_blob.resize(1, 1, 48, 48)

    return input_blob

def output(frame):
    preproc = prepare_sample(frame)
    opencv_net.setInput(preproc)
    out = opencv_net.forward()
    return label_decoder.inverse_transform([np.argmax(out, -1)])[0]

opencv_net = readNetFromONNX('necessary_files/resnext.onnx')
cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('necessary_files/haarcascade_frontalface_default.xml')
label_decoder = pickle.load(open('preproc_data/label_encoder.pkl', 'rb'))

path = input('enter path image: ')

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.putText(img, output(img[y:y + h, x:x + w]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('res_'+path, img)
cv2.imshow('RES', img)

cv2.waitKey(0)