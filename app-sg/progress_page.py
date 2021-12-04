import PySimpleGUI as sg
from utils import back_event


def progress_layout():
    BAR_MAX = 100
    layout = [[sg.Text('Wait while prediction is in progress', font=('Courier New', 25), pad=((0, 0), (60, 0)))],
              [sg.HSep(pad=((0, 0), (0, 26)))],
              [sg.ProgressBar(BAR_MAX, orientation='h', size=(40, 40), key='-PROGRESS BAR-')],
              [sg.Button('Cancel', size=(10, 1), font=('Courier New', 12))]]
    return layout


def progress_loop(window, chosen_stuff):
    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Back' in event:
            back_event(window)
            return


if __name__ == "__main__":
    layout = progress_layout()
    window = sg.Window("Progress Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
