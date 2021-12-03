import PySimpleGUI as sg
from settings_page import settings_layout

# from utils import ColumnFixedSize


class HECKApp:
    def __init__(self, *args, **kwargs):
        sg.theme("DefaultNoMoreNagging")
        sg.theme_button_color(('blue', '#b8cde0'))
        sg.set_options(font=('Courier New', 10))

        # ----------- Create all the layouts this Window will display -----------
        layout1 = [[sg.Text('Human Emotion Classification Kit', font=('Courier New', 25), pad=((0, 0), (70, 0)))],
                   [sg.HSep(pad=((0, 0), (0, 20)))],
                   [sg.Button('Prediction demo', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
                   [sg.Button('Load images', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
                   [sg.Button('Prediction settings', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))]]

        layout2 = [[sg.Text('Load Page interior')],
                   [sg.Button('Back')]]

        layout3 = settings_layout()

        layout4 = [[sg.Text('Prediction Page interior')],
                   [sg.Button('Back')]]

        # ----------- Create actual layout using Columns and a row of Buttons -----------
        layout = [[sg.Column(layout1, key='-COL1-', element_justification='center', vertical_alignment='c',
                             expand_y=True),
                   sg.Column(layout2, visible=False, key='-COL2-', element_justification='center'),
                   sg.Column(layout3, visible=False, key='-COL3-', element_justification='center'),
                   sg.Column(layout4, visible=False, key='-COL4-', element_justification='center')]]

        window = sg.Window('Human Emotion Classification Kit', layout, element_justification='center',
                           size=(800, 600), finalize=True)

        layout = 1
        while True:
            event, values = window.read()
            print(event, values)
            if event in (None, 'Exit'):
                break
            if event == 'Prediction demo':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL4-'].update(visible=True)
            if event == 'Load images':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL2-'].update(visible=True)
            if event == 'Prediction settings':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL3-'].update(visible=True)
            if 'Back' in event:
                window[f'-COL2-'].update(visible=False)
                window[f'-COL3-'].update(visible=False)
                window[f'-COL4-'].update(visible=False)
                window[f'-COL1-'].update(visible=True)
        window.close()


if __name__ == "__main__":
    app = HECKApp()
