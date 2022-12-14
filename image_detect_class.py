import pickle
import cv2
from cv2.dnn import readNetFromONNX
from necessary_files.preproc_sample import *
import argparse
import warnings

warnings.filterwarnings("ignore")

def processing_display_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.putText(img, output(img[y:y + h, x:x + w], opencv_net, label_decoder), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('res_' + path, img)
    cv2.imshow('RES', img)

    cv2.waitKey(0)

if __name__ == '__main__':
    opencv_net = readNetFromONNX('necessary_files/resnext.onnx')

    face_cascade = cv2.CascadeClassifier('necessary_files/haarcascade_frontalface_default.xml')
    label_decoder = pickle.load(open('preproc_data/label_encoder.pkl', 'rb'))

    parser = argparse.ArgumentParser(description='Face recognition and classification of emotions in an image.')
    parser.add_argument("--i", "--images", dest="images", nargs='+',
                        help="enter this argument if you need to enter more than one image")

    parser.add_argument('image', nargs="?", type=str, help='path to image')

    args = parser.parse_args()

    if args.image:
        processing_display_image(args.image)
    else:
        for path in args.images:
            processing_display_image(path)


