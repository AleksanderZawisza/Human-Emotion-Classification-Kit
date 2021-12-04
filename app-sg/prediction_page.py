import PySimpleGUI as sg
from utils import back_event


def prediction_layout():
    loaded_files_column = [[sg.Text("Currently loaded files/folders"),
                            ],
                           [sg.Listbox(values=[], enable_events=True, size=(42, 22), key="-LOADED FILES SHOW-",
                                       horizontal_scroll=True, highlight_background_color='#81b2db')
                            ],
                           ]

    chosen_files_column = [[sg.Frame('Files/folders chosen for prediction:',
                                     [[sg.Listbox(values=[], enable_events=True, size=(50, 8), key="-LOADED LIST-",
                                                  horizontal_scroll=True,
                                                  highlight_background_color='#81b2db')],
                                      [sg.Button('Unload selected', enable_events=True, key="-DELETE-")]],
                                     border_width=0, element_justification='center'), ],
                           [sg.Frame('Image preview', [[sg.Image(key="-IMAGE-")]],
                                     size=(400, 275), border_width=0, pad=(0, 0),
                                     element_justification='center')],
                           ]

    col1 = sg.Column(loaded_files_column, size=(370, 470), element_justification='center')
    col2 = sg.Column(chosen_files_column, pad=(20, 3), vertical_alignment='top', element_justification='center')

    layout = [[sg.Column([[sg.Text('Predict emotions from images', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 0)))]])],
              [sg.Frame("", [[col1,
                              # sg.VSep(),
                              col2]], pad=(0, 0), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center')], ]
    return layout


def predict_loop(window, loaded_stuff):
    while True:
        event, values = window.read()

        if event == 'Back':
            back_event(window)
            break

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == 'Predict emotions':
            window["-LOADED FILES SHOW-"].update(loaded_stuff)


if __name__ == "__main__":
    layout = prediction_layout()
    window = sg.Window("Base Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
