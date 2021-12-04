import PySimpleGUI as sg


def back_event(window):
    window[f'-COL2-'].update(visible=False)
    window[f'-COL3-'].update(visible=False)
    window[f'-COL4-'].update(visible=False)
    window[f'-COL1-'].update(visible=True)


def predict(window):
    event, values = window.read()

