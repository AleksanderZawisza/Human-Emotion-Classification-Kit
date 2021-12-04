import PySimpleGUI as sg
from utils import back_event
from PIL import Image
from io import BytesIO
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
                                              key="-LOADED FILES SHOW-",
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
                  sg.Button('Predict', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center'),
               ], ]
    return layout


def predict_loop(window, loaded_stuff):
    cwd = os.getcwd().replace('\\', '/')
    predpath = f"{cwd}/predictions"

    window["-LOADED FILES SHOW-"].update(loaded_stuff)
    window["-RESULT FOLDER-"].update(value=predpath)

    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Back' in event:
            back_event(window)
            return

        if event == "-LOADED FILES SHOW-" or event == '-CHOSEN FILE LIST-':  # A file was chosen from the listbox
            try:
                if event == "-LOADED FILES SHOW-":
                    filename = values["-LOADED FILES SHOW-"][0]
                if event == '-CHOSEN FILE LIST-':
                    filename = values["-CHOSEN FILE LIST-"][0]
                try:
                    im = Image.open(filename)
                except:
                    pass
                width, height = (350, 250)
                scale = max(im.width / width, im.height / height)
                if scale > 1:
                    w, h = int(im.width / scale), int(im.height / scale)
                    im = im.resize((w, h), resample=Image.CUBIC)
                with BytesIO() as output:
                    im.save(output, format="PNG")
                    data = output.getvalue()
                window["-IMAGE-"].update(data=data)
            except:
                pass


if __name__ == "__main__":
    layout = prediction_layout()
    window = sg.Window("Base Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
