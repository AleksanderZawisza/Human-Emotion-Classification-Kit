import PySimpleGUI as sg


def base_layout():
    layout = [[sg.Text('Human Emotion Classification Kit', font=('Courier New', 25), pad=((0, 0), (70, 0)))],
              [sg.HSep(pad=((0, 0), (0, 20)))],
              [sg.Button('Prediction demo', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
              [sg.Button('Load images', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
              [sg.Button('Prediction settings', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))]]
    return layout
