import PySimpleGUI as sg


def base_layout():
    layout = [[sg.Text('Human Emotion Classification Kit', font=('Courier New', 25), pad=((0, 0), (60, 0)))],
              [sg.HSep(pad=((0, 0), (0, 26)))],
              [sg.Button('Predict emotions', size=(30, 2), pad=((0, 0), (35, 0)), font=('Courier New', 14))],
              [sg.Button('Load images', size=(30, 2), pad=((0, 0), (35, 0)), font=('Courier New', 14))],
              [sg.Button('Train models', size=(30, 2), pad=((0, 0), (35, 0)), font=('Courier New', 14))],
              [sg.Button('Settings', size=(30, 2), pad=((0, 0), (35, 0)), font=('Courier New', 14))]]
    return layout


if __name__ == "__main__":
    layout = base_layout()
    window = sg.Window("Base Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
