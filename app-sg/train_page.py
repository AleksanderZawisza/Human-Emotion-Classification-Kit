import PySimpleGUI as sg
from utils import back_event
from PIL import Image
from io import BytesIO
import os


def train_layout():
    layout = [[sg.Column([[sg.Text('Change training settings', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 4)))]])],
              [sg.Frame('Model settings', [
                  [sg.Text('Choose model:')],
                  [sg.DropDown(['TensorFlow_ResNet9',
                                'TensorFlow_ResNet18',
                                'TensorFlow_ResNet34',
                                'TensorFlow_ResNet50',
                                'TensorFlow_ResNet101',
                                'TensorFlow_ResNet152',
                                'PyTorch_ResNet9',
                                'PyTorch_ResNet18',
                                'PyTorch_ResNet34',
                                'PyTorch_ResNet50',
                                'PyTorch_ResNet101',
                                'PyTorch_ResNet152',
                                ], key='-TRAIN DROPDOWN-',
                               background_color='#e3e3e3',
                               auto_size_text=True, expand_x=True, readonly=True, text_color='black',
                               enable_events=True)],
                  [sg.Frame('Optimizer settings:',
                           [[sg.Radio("SGD", group_id=3, default=True, key="-SGD-", enable_events=True,
                                      circle_color='blue'),
                             sg.Radio("Adam", group_id=3, key="-ADAM-", enable_events=True,
                                      circle_color='blue')]],
                            border_width=0,)],
                  [sg.Frame('', [[sg.Text('Learning rate', size=(15, 1)),
                                  sg.Slider((0.001, 0.3), orientation='horizontal', resolution=0.001, pad=((0, 0), (0, 5)),
                                            default_value=0.001, relief=sg.RELIEF_FLAT, trough_color='#e3e3e3',
                                            key="-LR-",
                                            size=(20, 16))],
                                 [sg.Text('Decay', size=(15, 1)),
                                  sg.Slider((0, 0.1), orientation='horizontal', resolution=0.001, pad=((0, 0), (0, 5)),
                                            default_value=0, relief=sg.RELIEF_FLAT, trough_color='#e3e3e3', key="-DECAY-",
                                            size=(20, 16))],
                                 [sg.Text('Model Save Name', size=(15, 1)),
                                  sg.Input(size=(20, 16), pad=((0, 0), (0, 5)), enable_events=True, key="-MODEL SAVE NAME-", )],
                                 ],
                            expand_x=True, expand_y=True, element_justification='center', border_width=0, pad=(0, 0),
                            ),
                   sg.Frame('', [[sg.Image(key="-FD IMAGE-")]],
                            expand_x=True, expand_y=True, border_width=0, pad=(0, 0),
                            element_justification='center')],
              ], expand_x=True, expand_y=True, border_width=0, font=('Courier New', 11))],

              [sg.Frame("",
                        [[
                            sg.Button("Back", enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=((0, 0), (16, 0)),
                        vertical_alignment='center')],
              ]
    return layout


def train_loop(window):
    while True:
        event, values = window.read()
        width, height = (390, 180)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if "Back" in event:
            back_event(window)
            return

        if event == "-NO FACE DETECTION-":
            window['-FACEDET DROPDOWN-'].update(disabled=True)
            window['-FD1-'].update(disabled=True)
            window['-FD2-'].update(disabled=True)
            window['-FD3-'].update(disabled=True)
            window['-FDSUB-'].update(disabled=True)
            window["-FD IMAGE-"].update(data=[])

        if event == "-FACE DETECTION-":
            window['-FACEDET DROPDOWN-'].update(disabled=False)
            window['-FD1-'].update(disabled=False)
            window['-FD2-'].update(disabled=False)
            window['-FD3-'].update(disabled=False)
            window['-FDSUB-'].update(disabled=False)
            event = '-FACEDET DROPDOWN-'

        if event == '-FACEDET DROPDOWN-':
            filepath = values['-FACEDET DROPDOWN-']
            if os.path.isdir(filepath):
                file_list = os.listdir(filepath)
                file_list = [f"{filepath}/{f}" for f in file_list
                             if os.path.isfile(f"{filepath}/{f}")]
                filepath = file_list[0]
            if os.path.isfile(filepath):
                try:
                    im = Image.open(filepath)
                except:
                    pass
                width, height = width, height
                scale = max(im.width / width, im.height / height)
                if scale > 1:
                    w, h = int(im.width / scale), int(im.height / scale)
                    im = im.resize((w, h), resample=Image.CUBIC)
                with BytesIO() as output:
                    im.save(output, format="PNG")
                    data = output.getvalue()
                window["-FD IMAGE-"].update(data=data)


if __name__ == "__main__":
    layout = train_layout()
    window = sg.Window("Train Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
