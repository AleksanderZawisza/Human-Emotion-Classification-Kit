import PySimpleGUI as sg
from settings_page import settings_layout, settings_loop
from base_page import base_layout
from load_page import load_layout, load_loop
from prediction_page import prediction_layout, predict_loop
from utils import back_event


class HECKApp:
    def __init__(self, *args, **kwargs):
        sg.theme("DefaultNoMoreNagging")
        sg.theme_button_color(('blue', '#b8cde0'))
        sg.set_options(font=('Courier New', 10))

        # ----------- Create all the layouts this Window will display -----------
        layout1 = base_layout()

        layout2 = load_layout()

        layout3 = settings_layout()

        layout4 = prediction_layout()

        # ----------- Create actual layout using Columns and a row of Buttons -----------
        layout = [[sg.Column(layout1, key='-COL1-', element_justification='center', vertical_alignment='c',
                             expand_y=True),
                   sg.Column(layout2, visible=False, key='-COL2-', element_justification='center', pad=(0, 0)),
                   sg.Column(layout3, visible=False, key='-COL3-', element_justification='center'),
                   sg.Column(layout4, visible=False, key='-COL4-', element_justification='center')]]

        window = sg.Window('Human Emotion Classification Kit', layout, element_justification='center',
                           size=(800, 600))

        layout = 1
        loaded_stuff = []
        while True:
            event, values = window.read()
            print(event, values)
            print(loaded_stuff)
            if event in (None, 'Exit'):
                break
            if event == 'Predict emotions':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL4-'].update(visible=True)
                predict_loop(window, loaded_stuff)
            if event == 'Load images':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL2-'].update(visible=True)
                loaded_stuff = load_loop(window, loaded_stuff)
            if event == 'Settings':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL3-'].update(visible=True)
                settings_loop(window)
            if 'Back' in event:
                back_event(window)
        window.close()


if __name__ == "__main__":
    app = HECKApp()
