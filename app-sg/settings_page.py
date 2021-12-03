import PySimpleGUI as sg


def settings_layout():
    layout = [[sg.Text('Change prediction settings', font=('Courier New', 20))],
              [sg.Frame('Choose model',
                        [[sg.Radio("Model 1", group_id=1, default=True)],
                         [sg.Radio("Model 2", group_id=1)],
                         [sg.Radio("Model 3", group_id=1)]], expand_x=True, pad=((0, 0), (50, 0)),
                        font=('Courier New', 12))],
              [sg.Frame('Use face detection?',
                        [[sg.Radio("Yes", group_id=2, default=True)],
                         [sg.Radio("No", group_id=2)]], expand_x=True, pad=((0, 0), (50, 0)),
                        font=('Courier New', 12))],
              [sg.Button('Back', pad=((0, 0), (50, 0)), size=(10, 1), font=('Courier New', 14))]]
    return layout
