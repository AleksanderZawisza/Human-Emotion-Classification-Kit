import PySimpleGUI as sg
import cv2

def back_event(window):
    window[f'-COL2-'].update(visible=False)
    window[f'-COL3-'].update(visible=False)
    window[f'-COL4-'].update(visible=False)
    window[f'-COL1-'].update(visible=True)


def predict(window):
    event, values = window.read()


def simple_detect_draw_face(img_path, save_dir, faceCascade, scale, minneigh, minsize):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=int(minneigh),
        minSize=(int(minsize), int(minsize)),
    )

    #img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #img = img[y:y + h, x:x + w]

    cv2.imwrite(save_dir, img)



