import os

import cv2
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import dlib
from keras.models import load_model


def back_event(window):
    window[f'-COL2-'].update(visible=False)
    window[f'-COL3-'].update(visible=False)
    window[f'-COL4-'].update(visible=False)
    window[f'-COL7-'].update(visible=False)
    window[f'-COL1-'].update(visible=True)


def create_result_text_folder(result_list, chosen_path):
    # print(result_list)
    image_names = os.listdir(chosen_path)
    results = [0, 0, 0, 0, 0, 0, 0]
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    for k in range(7):
        n = 0
        for f in image_names:
            file_path = f'{chosen_path}/{f}'
            for j in range(len(result_list[file_path])):
                results[k] += result_list[file_path][j][k]
                n += 1
        results[k] = results[k] / n
    for i, key in enumerate(emotions_dict):
        emotions_dict[key] = results[i]
        sorted_keys = sorted(emotions_dict, key=emotions_dict.get, reverse=True)
        emotion_string = ""
    for key in sorted_keys:
        rounded = round(emotions_dict[key], 1)
        newline = f"{key}: {rounded}%\n"
        emotion_string = emotion_string + newline
    emotion_string = emotion_string[:-1]  # wywalenie ostatniego entera
    return emotion_string


def create_result_text(result_list):
    # print(result_list)
    results = [0, 0, 0, 0, 0, 0, 0]
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    for k in range(7):
        for j in range(len(result_list)):
            results[k] += result_list[j][k]
        results[k] = results[k] / len(result_list)
    for i, key in enumerate(emotions_dict):
        emotions_dict[key] = results[i]
        sorted_keys = sorted(emotions_dict, key=emotions_dict.get, reverse=True)
        emotion_string = ""
    for key in sorted_keys:
        rounded = round(emotions_dict[key], 1)
        newline = f"{key}: {rounded}%\n"
        emotion_string = emotion_string + newline
    emotion_string = emotion_string[:-1]  # wywalenie ostatniego entera
    return emotion_string


def simple_detect_draw_face(img_path, save_dir, faceCascade, scale, minneigh, minsize):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=int(minneigh),
        minSize=(int(minsize), int(minsize)),
    )

    # img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # img = img[y:y + h, x:x + w]

    cv2.imwrite(save_dir, img)


def write_emotions_on_img(img, emotion_preds, bottomLeftCornerOfText, width, faceid=0, topLeftCorner=(0, 0)):
    fontpath = "Lato-Semibold.ttf"
    img_pil = Image.fromarray(img)
    height = bottomLeftCornerOfText[1] - topLeftCorner[1]
    biggness = min(width, height)
    fontsize = biggness // 8
    # fontsize = min(fontsize, 40)
    fontsize = max(fontsize, 7)
    font = ImageFont.truetype(fontpath, fontsize)

    if fontsize == 7:
        xstep = 1
        up = 1
    else:
        xstep = 5
        up = 5

    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    for i, key in enumerate(emotions_dict):
        emotions_dict[key] = emotion_preds[i]
    sorted_keys = sorted(emotions_dict, key=emotions_dict.get, reverse=True)
    emotion_string = ""
    upstep = fontsize + 2
    for key in sorted_keys:
        if emotions_dict[key] > 20:
            up += upstep
            rounded = round(emotions_dict[key], 1)
            newline = f"{key}: {rounded}%\n"
            emotion_string = emotion_string + newline
    emotion_string = emotion_string[:-1]  # wywalenie ostatniego entera

    draw = ImageDraw.Draw(img_pil)
    draw.text((topLeftCorner[0] + xstep, topLeftCorner[1]), f"id: {faceid}", font=font, fill=(0, 0, 255, 0))

    y0, dy = bottomLeftCornerOfText[1] - up, upstep
    for i, line in enumerate(emotion_string.split('\n')):
        y = y0 + i * dy
        draw.text((bottomLeftCornerOfText[0] + xstep, y), line, font=font,
                  fill=(0, 0, 255, 0))

    img = np.array(img_pil)

    return img


def prediction_combo(img_path, save_dir, model, model_text, detection, faceCascade, scale, minneigh, minsize,
                     predictor=[]):
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = []
    preds = []
    if detection:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=int(minneigh),
            minSize=(int(minsize), int(minsize)),
        )

        # img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(faces):
            img_tmp = img[y:y + h, x:x + w]

            if model_text == '-RESNET9-' or "PyTorch" in model_text:
                out = predict_res9pt(img_tmp, model)
            else:
                out = predict_res50tf(img_tmp, model, predictor)

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bottomLeftCornerOfText = (x, y + h)
            topLeftCorner = (x, y)
            img = write_emotions_on_img(img, out, bottomLeftCornerOfText, w, i, topLeftCorner)
            preds.append(out)

    if not detection or len(faces) == 0:
        if model_text == '-RESNET9-' or "PyTorch" in model_text:
            out = predict_res9pt(img, model)
        else:
            out = predict_res50tf(img, model, predictor)

        bottomLeftCornerOfText = (0, img.shape[0])
        img = write_emotions_on_img(img, out, bottomLeftCornerOfText, img.shape[1])
        preds.append(out)

    res, pic_name = os.path.split(img_path)
    save_path = f'{save_dir}/{pic_name}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_path, img)

    result_dict = {save_path: preds}
    change_dict = {save_path: img_path}
    return result_dict, change_dict


def list_all_pictures(chosen_stuff):
    pic_list = []
    for entity in chosen_stuff:
        if os.path.isdir(entity):
            for f in os.listdir(entity):
                if os.path.isfile(f"{entity}/{f}") and f.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    pic_list.append(f"{entity}/{f}")
        else:
            pic_list.append(entity)
    return pic_list


def load_res9pt():
    cwd = os.getcwd().replace('\\', '/')
    model_path = f"{cwd}/models/ResNet1_mdl_EPOCHS_40.pth"
    return torch.load(model_path)


def load_res50tf():
    model_path = "models/RESNET50-MODYFIKACJA-EPOCHS_30test_acc_0.681.h5"
    return load_model(model_path)


def load_custom_model(model_text):
    model_path = "user_models/" + model_text
    if "PyTorch" in model_text:
        return torch.load(model_path)
    else:
        return load_model(model_path)


def predict_res9pt(img, model):
    img = np.asarray(img)

    preprocess = tt.Compose([tt.Resize((64, 64)),
                             tt.Grayscale(num_output_channels=1),
                             tt.ToTensor()])

    img_preprocessed = preprocess(Image.fromarray(img))
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    out = model(batch_img_tensor)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100  # procenty
    return percentage.tolist()  # normalna pythonowa lista


def facial_landmarks(image, predictor):
    # image = cv2.imread(filepath)
    face_rects = [dlib.rectangle(left=1, top=1, right=len(image) - 1, bottom=len(image) - 1)]
    face_landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face_rects[0]).parts()])
    return face_landmarks


def predict_res50tf(img, model, predictor):
    X1 = []
    X2 = []
    resize = 197

    # img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (resize, resize))
    features = facial_landmarks(img, predictor)
    img = img / 255

    X1.append(img)
    X2.append(features)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X = [X1, X2]
    Y_pred = model.predict(X)[0]
    Y_pred = Y_pred * 100  # procenty
    return Y_pred.tolist()  # normalna pythonowa lista


if __name__ == "__main__":
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    image_path = "example_images/sad1.png"
    img = cv2.imread(image_path)
    # start = time.time()
    # predictor = dlib.shape_predictor('faceutils/shape_predictor_68_face_landmarks.dat')
    # model = load_res50tf()
    # end = start - time.time()
    # pred = predict_res50tf(img, model, predictor)
    model = load_res9pt()
    pred = predict_res9pt(img, model)
    img = write_emotions_on_img(img, pred, (0, img.shape[0]), img.shape[1])
    cv2.imwrite('test.png', img)
# print(pred)
