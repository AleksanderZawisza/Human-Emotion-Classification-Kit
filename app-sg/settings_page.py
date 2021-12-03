import PySimpleGUI as sg


def settings_layout():
    layout = [[sg.Text('Change prediction settings', font=('Courier New', 20))],
              [sg.Frame('Choose model',
                        [[sg.Radio("ResNet50 modified TensorFlow", group_id=1, default=True)],
                         [sg.Radio("VGG16 modified TensorFlow", group_id=1)],
                         [sg.Radio("ResNet9 PyTorch", group_id=1)]], expand_x=True, pad=((0, 0), (50, 0)),
                        font=('Courier New', 12))],
              [sg.Frame('Use face detection?',
                        [[sg.Radio("Yes", group_id=2, default=True)],
                         [sg.Radio("No", group_id=2)]], expand_x=True, pad=((0, 0), (50, 0)),
                        font=('Courier New', 12))],
              [sg.Button('Back', pad=((0, 0), (50, 0)), size=(10, 1), font=('Courier New', 12))]]
    return layout


if __name__ == "__main__":
    layout = settings_layout()
    window = sg.Window("Settings Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
