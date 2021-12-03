import PySimpleGUI as sg
import os.path
from PIL import Image
from io import BytesIO
from utils import BackEvent

def load_layout():
    # sg.theme("DefaultNoMoreNagging")
    # sg.theme_button_color((('blue', '#b8cde0')))
    # sg.set_options(font=('Courier New', 10))

    # First the window layout in 2 columns
    file_list_column = [[sg.Text("Folder"),
                         sg.In(size=(28, 1), enable_events=True, key="-FOLDER-"),
                         sg.FolderBrowse(),
                         ],
                        [sg.Listbox(values=[], enable_events=True, size=(42, 27), key="-FILE LIST-",
                                    horizontal_scroll=True, highlight_background_color='#81b2db')
                        ],
                        [sg.Button('Load Folder', enable_events=True, key="-LOAD FOLDER-"),
                         sg.Button('Load Image', enable_events=True, key="-LOAD IMAGE-"),]
                        ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [[sg.Frame('Loaded folders/files:',
                                     [[sg.Listbox(values=[], enable_events=True, size=(50, 8), key="-LOADED LIST-",
                                                  horizontal_scroll=True,
                                       highlight_background_color='#81b2db')],
                                      [sg.Button('Unload selected', enable_events=True, key="-DELETE-")]],
                                     border_width=0, element_justification='center'),],
                           [sg.Frame('Image preview',[[sg.Image(key="-IMAGE-")]],
                                     size=(400, 320), border_width=0, pad=(0, 0),
                                     element_justification='center')],
                           ]

    col1 = sg.Column(file_list_column, size=(370,550), element_justification='center')
    col2 = sg.Column(image_viewer_column, pad=(20,3), vertical_alignment='top', element_justification='center')

    # ----- Full layout -----
    layout = [[sg.Frame("",[[col1,
               sg.VSeperator(),
               col2]], pad=(0,0), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10,1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='bottom') ],]

    return layout

def load_loop(window, loaded_stuff):
    # window = sg.Window("Dataset/Image Loader", layout, element_justification='center',
    #                    size=(800, 600))

    # Run the Event Loop
    while True:
        event, values = window.read()

        if event == "Back":
            BackEvent(window)
            return loaded_stuff

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            # WYSWIETLANIE ZDJEC Z PIL
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [f for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-" or event == '-LOADED LIST-':  # A file was chosen from the listbox
            try:
                if event == "-FILE LIST-":
                    filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                if event == '-LOADED LIST-':
                    filename = values["-LOADED LIST-"][0]
                try:
                    im = Image.open(filename)
                except:
                    pass
                width, height = (340, 300)
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

        elif event == "-LOAD FOLDER-":
            try:
                folder = values["-FOLDER-"]
                if os.path.isdir(folder) and folder not in loaded_stuff:
                    loaded_stuff.append(folder)
                    window["-LOADED LIST-"].update(loaded_stuff)
            except:
                pass

        elif event == "-LOAD IMAGE-":
            try:
                folder = values["-FOLDER-"]
                file = values["-FILE LIST-"][0]
                filename = f"{folder}/{file}"
                if os.path.isfile(filename) and filename not in loaded_stuff:
                    loaded_stuff.append(filename)
                    window["-LOADED LIST-"].update(loaded_stuff)
            except:
                pass

        elif event == "-DELETE-":
            try:
                stuff_to_delete = values["-LOADED LIST-"][0]
                if stuff_to_delete in loaded_stuff:
                    loaded_stuff.remove(stuff_to_delete)
                    window["-LOADED LIST-"].update(loaded_stuff)
            except:
                pass

if __name__ == "__main__":
    layout = load_layout()
    window = sg.Window("Dataset/Image Loader", layout, element_justification='center',
                        size=(800, 600))
    load_loop(window)