import PySimpleGUI as sg
from utils import back_event
import os
import cv2

def settings_layout():
    layout = [[sg.Column([[sg.Text('Change prediction settings', font=('Courier New', 20))],
              [sg.HSep(pad=((0, 0), (0, 4)))]])],
              [sg.Frame('Choose model',
                        [[sg.Text('Accuracy: 69% | Face Det. Accuracy: 75% | Face Pred. Time: 0.15s',
                                  font=('Courier 10'))],
                         [sg.Radio("ResNet50 modified TensorFlow", group_id=1, default=True, key="-RESNET50-")],

                         # [sg.HSep(pad=(50, 10))],
                         # [sg.Text('Accuracy: 60% | Face Det. Accuracy: 74% | Face Pred. Time: 0.20s',
                         #          font=('Courier 10'))],
                         # [sg.Radio("VGG16 modified TensorFlow", group_id=1)],

                         [sg.HSep(pad=(50, 10))],
                         [sg.Text('Accuracy: 63% | Face Det. Accuracy: 73% | Face Pred. Time: 0.09s',
                                  font=('Courier 10'))],
                         [sg.Radio("ResNet9 PyTorch", group_id=1, key="-RESNET9-")],

                         ],
                        expand_x=True, pad=((0, 0), (8, 0)), size=(560, 160),
                        font=('Courier New', 12), element_justification='center')],
              [sg.Frame('Use Face Detection?',
                        [[sg.Radio("Yes", group_id=2, default=True, key="-FACE DETECTION-", enable_events=True),
                         sg.Radio("No", group_id=2, key="-NO FACE DETECTION-", enable_events=True)],
                         [sg.Frame('Face Detection settings', [
                             [sg.Text('Choose folder/image for preview:')],
                             [sg.DropDown(['Load images to get Face Detection preview'], key='-FACEDET DROPDOWN-',
                                          auto_size_text=True, expand_x=True, readonly=True, text_color='black', enable_events=True)],
                             [sg.Frame('', [[sg.Slider((0.1,3), orientation='horizontal', resolution=0.1, pad=(0,0),
                                                       default_value=1.1, relief=sg.RELIEF_FLAT, trough_color='#b8cde0')]],
                                     expand_x=True, expand_y=True, border_width=1, pad=(0, 0),
                                     element_justification='center'),
                                 sg.Frame('', [[sg.Image(key="-FD IMAGE-")]],
                                       expand_x=True, expand_y=True, border_width=1, pad=(0, 0),
                                       element_justification='center')],
                         ], expand_x=True, expand_y=True, border_width=0, font=('Courier New', 11))]],
                        expand_x=True, pad=((0, 0), (5, 0)), size=(560, 300),
                        font=('Courier New', 12), element_justification="center")], #Gdzies kolo tego dac ustawienia do face detection
              [sg.Frame("",
                        [[
                  sg.Button("Back", enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=((0, 0), (16, 0)),
                        vertical_alignment='center')],
              ]
    return layout

def settings_loop(window, loaded_stuff, faceCascade):

    if loaded_stuff:
        window['-FACEDET DROPDOWN-'].update(values=loaded_stuff)
        window['-FACEDET DROPDOWN-'].expand()

    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if "Back" in event:
            back_event(window)
            return

        if event == "-NO FACE DETECTION-":
            window['-FACEDET DROPDOWN-'].update(disabled=True)

        if event == "-FACE DETECTION-":
            window['-FACEDET DROPDOWN-'].update(disabled=False)

        if event == '-FACEDET DROPDOWN-':
            file = values['-FACEDET DROPDOWN-']
            if os.path.isdir(file):
                file_list = os.path.listdir(file)
                file_list = [f for f in file_list
                      if os.path.isfile(f)
                      and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
                file = file_list[0]
            if os.path.isfile(file):
                pass








if __name__ == "__main__":
    layout = settings_layout()
    window = sg.Window("Settings Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
