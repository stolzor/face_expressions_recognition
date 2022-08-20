import pickle
import cv2
from cv2.dnn import readNetFromONNX
from necessary_files.preproc_sample import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    opencv_net = readNetFromONNX('necessary_files/resnext.onnx')
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('necessary_files/haarcascade_frontalface_default.xml')
    label_decoder = pickle.load(open('preproc_data/label_encoder.pkl', 'rb'))
    c = 0
    while True:
        flag, frame = cap.read()

        if flag:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            print(faces)

            for (x, y, w, h) in faces:
                cv2.putText(frame, output(frame[y:y + h, x:x + w], opencv_net, label_decoder), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) == ord('q'):
            break
        c += 1
    cap.release()
    cv2.destroyAllWindows()





