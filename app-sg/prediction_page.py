import PySimpleGUI as sg


def prediction_layout():
    loaded_files_column = [[sg.Text("Currently loaded files/folders"),
                            ],
                           [sg.Listbox(values=[], enable_events=True, size=(42, 22), key="-LOADED LIST-",
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

    layout = [[sg.Text('Predict emotions from images', font=('Courier New', 25), pad=((0, 0), (60, 0)))],
              [sg.HSep(pad=((0, 0), (0, 26)))],
              [sg.Button('Predict emotions', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
              [sg.Button('Load images', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))],
              [sg.Button('Settings', size=(30, 2), pad=((0, 0), (50, 0)), font=('Courier New', 14))]]
    return layout


if __name__ == "__main__":
    layout = prediction_layout()
    window = sg.Window("Base Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
