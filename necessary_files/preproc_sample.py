import cv2
import numpy as np

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

def output(frame, model, decoder):
    preproc = prepare_sample(frame)
    decoder.setInput(preproc)
    out = model.forward()
    return decoder.inverse_transform([np.argmax(out, -1)])[0]