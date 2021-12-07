import PySimpleGUI as sg
from utils import back_event
from progress_page import progress_loop
import os


def prediction_layout():
    choosing_column = [[sg.Frame('Choose location to save prediction results:',
                                 [[sg.Text("Folder"),
                                   sg.In(size=(72, 2), enable_events=True, key="-RESULT FOLDER-"),
                                   sg.FolderBrowse()]],
                                 size=(800, 65), element_justification='left',
                                 border_width=0, pad=(0, 2))],
                       [sg.Frame('Choose files/folders for prediction:',
                                 [[sg.Listbox(values=[], enable_events=True, size=(87, 30),
                                              key="-CHOOSE FILES-",
                                              horizontal_scroll=True, highlight_background_color='#81b2db',
                                              select_mode="multiple")
                                   ]],
                                 size=(800, 400), border_width=0, pad=(0, 0),
                                 element_justification='left')],
                       ]

    col1 = sg.Column(choosing_column, size=(800, 470), pad=(20, 3),
                     element_justification='center')

    layout = [[sg.Column([[sg.Text('Predict emotions from images', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 0)))]])],
              [sg.Frame("", [[col1]], pad=(0, 10), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12)),
                  sg.Button('Predict', enable_events=True, size=(10, 1), font=('Courier New', 12), key='-PREDICT-')]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center'),
               ], ]
    return layout


def predict_loop(window, loaded_stuff, faceCascade, models, predictor):
    cwd = os.getcwd().replace('\\', '/')
    pred_path = f"{cwd}/predictions"
    example_path = f"{cwd}/example_images"

    # when user didnt load any images to predict on (get defaults)
    if not loaded_stuff:
        loaded_stuff = [f"{example_path}/angry1.png", f"{example_path}/sad1.png",
                        f"{example_path}/happy1.png", f"{example_path}/crowd.jpg"]

    chosen_stuff = []

    window["-CHOOSE FILES-"].update(loaded_stuff)
    window["-RESULT FOLDER-"].update(value=pred_path)

    while True:
        event, values = window.read()
        print(event, values)

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            return models, predictor

        if 'Back' in event:
            back_event(window)
            return models, predictor

        if event == '-CHOOSE FILES-':
            chosen_stuff = values['-CHOOSE FILES-']

        if event == "-PREDICT-":
            # print(chosen_stuff)
            if not chosen_stuff:
                sg.PopupOK("You have not chosen any images to predict on!", title='YOU SHALL NOT PASS!!!')
                continue
            window[f'-COL4-'].update(visible=False)
            window[f'-COL5-'].update(visible=True)
            models, predictor = progress_loop(window, chosen_stuff, values, faceCascade, models, predictor)


if __name__ == "__main__":
    layout = prediction_layout()
    window = sg.Window("Base Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
