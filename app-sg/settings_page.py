import PySimpleGUI as sg


def settings_layout():
    layout = [[sg.Column([[sg.Text('Change prediction settings', font=('Courier New', 20))],
              [sg.HSep(pad=((0, 0), (0, 4)))]])],
              [sg.Frame('Choose model',
                        [[sg.Text('Accuracy: 69% | Face Det. Accuracy: 75% | Face Pred. Time: 0.15s',
                                  font=('Courier 10'))],
                         [sg.Radio("ResNet50 modified TensorFlow", group_id=1, default=True)],

                         [sg.HSep(pad=(50, 10))],
                         [sg.Text('Accuracy: 60% | Face Det. Accuracy: 74% | Face Pred. Time: 0.20s',
                                  font=('Courier 10'))],
                         [sg.Radio("VGG16 modified TensorFlow", group_id=1)],

                         [sg.HSep(pad=(50, 10))],
                         [sg.Text('Accuracy: 63% | Face Det. Accuracy: 73% | Face Pred. Time: 0.09s',
                                  font=('Courier 10'))],
                         [sg.Radio("ResNet9 PyTorch", group_id=1)],

                         ],
                        expand_x=True, pad=((0, 0), (10, 0)), size=(560, 240),
                        font=('Courier New', 12), element_justification='center')],
              [sg.Frame('Use Face Detection?',
                        [[sg.Radio("Yes", group_id=2, default=True),
                         sg.Radio("No", group_id=2)]], expand_x=True, pad=((0, 0), (50, 0)),
                        font=('Courier New', 12), element_justification="center")], #Gdzies kolo tego dac ustawienia do face detection
              [sg.Button('Back', pad=((0, 0), (40, 0)), size=(10, 1), font=('Courier New', 12))]]
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
