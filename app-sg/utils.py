import os
import time

import cv2
import torch
import torch.nn as nn
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
    window[f'-COL1-'].update(visible=True)


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

def write_emotions_on_img(img, emotion_preds, bottomLeftCornerOfText, faceid=0, topLeftCorner=(0,0)):
    fontpath = "Lato-Semibold.ttf"
    img_pil = Image.fromarray(img)
    fontsize = min((bottomLeftCornerOfText[1] - topLeftCorner[1]) // 8, 32)
    fontsize = max(fontsize, 6)
    font = ImageFont.truetype(fontpath, fontsize)

    xstep = 5

    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    for i, key in enumerate(emotions_dict):
        emotions_dict[key] = emotion_preds[i]
    sorted_keys = sorted(emotions_dict, key=emotions_dict.get, reverse=True)
    emotion_string = ""
    upstep = fontsize+2
    up=5
    for key in sorted_keys:
        if emotions_dict[key]>20:
            up+=upstep
            rounded = round(emotions_dict[key], 1)
            newline = f"{key}: {rounded}%\n"
            emotion_string = emotion_string + newline
    emotion_string = emotion_string[:-1] #wywalenie ostatniego entera

    draw = ImageDraw.Draw(img_pil)
    draw.text((topLeftCorner[0]+xstep, topLeftCorner[1]), f"id: {faceid}", font=font, fill=(0, 0, 255, 0))

    y0, dy = bottomLeftCornerOfText[1]-up, upstep
    for i, line in enumerate(emotion_string.split('\n')):
        y = y0 + i * dy
        draw.text((bottomLeftCornerOfText[0]+xstep, y), line, font=font,
                  fill=(0, 0, 255, 0))

    img = np.array(img_pil)

    return img



def prediction_combo(img_path, save_dir, model, model_text, detection, faceCascade, scale, minneigh, minsize,
                     predictor=[]):
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = []
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

            if model_text == '-RESNET9-':
                out = predict_res9pt(img_tmp, model)
            else:
                out = predict_res50tf(img_tmp, model, predictor)

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bottomLeftCornerOfText = (x, y + h)
            topLeftCorner = (x, y)
            img = write_emotions_on_img(img, out, bottomLeftCornerOfText, i, topLeftCorner)


    if not detection or len(faces)==0:
        if model_text == '-RESNET9-':
            out = predict_res9pt(img, model)
        else:
            out = predict_res50tf(img, model, predictor)

        bottomLeftCornerOfText = (0, img.shape[0])
        img = write_emotions_on_img(img, out, bottomLeftCornerOfText)

    res, pic_name = os.path.split(img_path)
    save_path = os.path.join(save_dir, pic_name)
    cv2.imwrite(save_path, img)
    return model, predictor


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


def predict_res9pt(img, model):
    img = np.asarray(img)

    preprocess = tt.Compose([tt.Resize((64, 64)),
                             tt.Grayscale(num_output_channels=1),
                             tt.ToTensor()])

    img_preprocessed = preprocess(Image.fromarray(img))
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    out = model(batch_img_tensor)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #procenty
    return percentage.tolist() #normalna pythonowa lista


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
    Y_pred = Y_pred*100 #procenty
    return Y_pred.tolist()  #normalna pythonowa lista


# HELPER FUNCTIONS FOR MODEL LOADING AND PREDICTION

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


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
    img = write_emotions_on_img(img, pred, (0, img.shape[1]))
    cv2.imwrite('test.png', img)
    print(pred)

