import logging
logging.getLogger('tensorflow').disabled = True
import warnings
import cv2
import dlib
from utils import *

def model_pt_load_predict_test():
    image_path = "example_images/happy1.png"
    img = cv2.imread(image_path)
    model = load_res9pt()
    if model:
        print("PASS: Load PT model")
    else:
        raise NameError('FAIL: PT model not loaded')
    pred = predict_res9pt(img, model)
    if pred:
        print("PASS: Use PT model")
    else:
        raise NameError('FAIL: PT model does not return predictions')
    if isinstance(pred, list) and len(pred)==7 and round(sum(pred))==100:
        for p in pred:
            if not p>=0 and p<=100:
                raise ValueError('FAIL: PT model predictions negative or greater than 100')
        print('PASS: TF model returns list of percentage of 7 emotions')
    else:
        raise NameError('FAIL: PT model gives bad predictions')
    return pred

def model_tf_load_predict_test():
    image_path = "example_images/happy1.png"
    img = cv2.imread(image_path)
    predictor = dlib.shape_predictor('faceutils/shape_predictor_68_face_landmarks.dat')
    model = load_res50tf()
    if model:
        print("PASS: Load TF model")
    else:
        raise NameError('FAIL: TF model not loaded')
    pred = predict_res50tf(img, model, predictor)
    if pred:
        print("PASS: Use TF model")
    else:
        raise NameError('FAIL: TF model does not return predictions')
    if isinstance(pred, list) and len(pred)==7 and round(sum(pred))==100:
        for p in pred:
            if not p>=0 and p<=100:
                raise ValueError('FAIL: TF model predictions negative or greater than 100')
        print('PASS: TF model returns list of percentage of 7 emotions')
    else:
        raise NameError('FAIL: TF model gives bad predictions')
    return pred




if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model_pt_load_predict_test()
        model_tf_load_predict_test()