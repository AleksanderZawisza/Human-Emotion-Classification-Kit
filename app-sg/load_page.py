import PySimpleGUI as sg
import os.path
from PIL import Image
from io import BytesIO
from utils import back_event


def load_layout():
    # First the window layout in 2 columns
    file_list_column = [[sg.Text("Folder"),
                         sg.In(size=(28, 1), enable_events=True, key="-FOLDER-", readonly=True, focus=False,
                               disabled_readonly_background_color='white'),
                         sg.FolderBrowse(),
                         ],
                        [sg.Text("Filter"), sg.Input(size=(28, 1), enable_events=True, key="-FILTER-",
                                                     pad=((0, 70), (0, 0)))],
                        [sg.Button('Load Image', enable_events=True, key="-LOAD IMAGE-"),
                         sg.Button('Load Filtered', enable_events=True, key="-LOAD FILTERED-"),
                         sg.Button('Load Folder', enable_events=True, key="-LOAD FOLDER-"),
                         ],
                        [sg.Listbox(values=[], enable_events=True, size=(42, 21), key="-FILE LIST-",
                                    horizontal_scroll=True, highlight_background_color='#81b2db')
                         ],
                        ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [[sg.Frame('Loaded folders/files:',
                                     [
                                         [sg.Listbox(values=[], enable_events=True, size=(50, 8), key="-LOADED LIST-",
                                                     horizontal_scroll=True,
                                                     highlight_background_color='#81b2db')],
                                         [sg.Button('Unload selected', enable_events=True, key="-DELETE-"),
                                          sg.Button('Unload all', enable_events=True, key="-DELETE ALL-"), ],

                                     ],
                                     border_width=0, element_justification='center'), ],
                           [sg.Frame('Image preview', [[sg.Image(key="-IMAGE-")]],
                                     size=(400, 275), border_width=0, pad=(0, 0),
                                     element_justification='center')],
                           ]

    col1 = sg.Column(file_list_column, size=(370, 475), element_justification='center')
    col2 = sg.Column(image_viewer_column, pad=(20, 3), vertical_alignment='top', element_justification='center')

    # ----- Full layout -----
    layout = [[sg.Column([[sg.Text('Load images', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 0)))]])],
              [sg.Frame("", [[col1,
                              # sg.VSep(),
                              col2]], pad=(0, 0), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center')], ]

    return layout


def load_loop(window, loaded_stuff):
    # Run the Event Loop
    while True:
        event, values = window.read()
        # print(event, values)
        # print(window['-FILE LIST-'].get_list_values())

        if event == "Exit" or event == sg.WIN_CLOSED:
            return loaded_stuff

        if 'Back' in event:
            back_event(window)
            return loaded_stuff

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
                      and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            window["-FILE LIST-"].update(fnames)
            event = "-FILTER-"
            if len(fnames) == 0:
                sg.PopupOK('No images found in folder!', title='SORRY')

        if event == "-FILTER-":
            if values['-FOLDER-']:
                folder = values["-FOLDER-"]
                # WYSWIETLANIE ZDJEC Z PIL
                try:
                    file_list = os.listdir(folder)
                except:
                    file_list = []

                fnames = [f for f in file_list
                          if os.path.isfile(os.path.join(folder, f))
                          and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                currentFolderFiles = fnames
                if currentFolderFiles:
                    search = values['-FILTER-']
                    new_values = [x for x in currentFolderFiles if search.lower() in x.lower()]  # do the filtering
                    window['-FILE LIST-'].update(new_values)  # display in the listbox

        if event == "-FILE LIST-" or event == '-LOADED LIST-':  # A file was chosen from the listbox
            try:
                if event == "-FILE LIST-":
                    filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
                if event == '-LOADED LIST-':
                    filename = values["-LOADED LIST-"][0]
                if os.path.isdir(filename):
                    file_list = os.listdir(filename)
                    file_list = [f"{filename}/{f}" for f in file_list
                                 if os.path.isfile(f"{filename}/{f}")
                                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                    filename = file_list[0]
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

        if event == "-LOAD FOLDER-":
            if values['-FOLDER-']:
                folder = values["-FOLDER-"]
                try:
                    file_list = os.listdir(folder)
                except:
                    file_list = []
                fnames = [f for f in file_list
                          if os.path.isfile(os.path.join(folder, f))
                          and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                currentFolderFiles = fnames
                if not currentFolderFiles:
                    continue
            try:
                folder = values["-FOLDER-"]
                if os.path.isdir(folder) and folder not in loaded_stuff:
                    loaded_stuff.append(folder)
                    window["-LOADED LIST-"].update(loaded_stuff)
            except:
                pass

        if event == "-LOAD FILTERED-":
            filteredFiles = window['-FILE LIST-'].get_list_values()
            if filteredFiles:
                try:
                    folder = values["-FOLDER-"]
                    for file in filteredFiles:
                        filename = f"{folder}/{file}"
                        if os.path.isfile(filename) and filename not in loaded_stuff:
                            loaded_stuff.append(filename)
                            window["-LOADED LIST-"].update(loaded_stuff)
                except:
                    pass

        if event == "-LOAD IMAGE-":
            try:
                folder = values["-FOLDER-"]
                file = values["-FILE LIST-"][0]
                filename = f"{folder}/{file}"
                if os.path.isfile(filename) and filename not in loaded_stuff:
                    loaded_stuff.append(filename)
                    window["-LOADED LIST-"].update(loaded_stuff)
            except:
                pass

        if event == "-DELETE-":
            try:
                stuff_to_delete = values["-LOADED LIST-"][0]
                if stuff_to_delete in loaded_stuff:
                    loaded_stuff.remove(stuff_to_delete)
                    window["-LOADED LIST-"].update(loaded_stuff)
                    window["-IMAGE-"].update(data=[])
            except:
                pass

        if event == "-DELETE ALL-":
            try:
                loaded_stuff = []
                window["-LOADED LIST-"].update(loaded_stuff)
                window["-IMAGE-"].update(data=[])
            except:
                pass


if __name__ == "__main__":
    layout = load_layout()
    window = sg.Window("Dataset/Image Loader", layout, element_justification='center',
                       size=(800, 600))
    loaded_stuff = []
    load_loop(window, loaded_stuff)
