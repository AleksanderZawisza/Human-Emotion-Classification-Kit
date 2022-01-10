import PySimpleGUI as sg
from utils import back_event
from PIL import Image
from io import BytesIO
from train_page import train_loop
import os


def train_settings_layout():
    layout = [[sg.Column([[sg.Text('Change training settings', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 46)))]])],
              [sg.Frame('Model settings', [
                  [sg.Text('Choose model:')],
                  [sg.DropDown(['TensorFlow_ResNet9',
                                'TensorFlow_ResNet18',
                                'TensorFlow_ResNet34',
                                'TensorFlow_ResNet50',
                                'TensorFlow_ResNet101',
                                'TensorFlow_ResNet152',
                                'PyTorch_ResNet9',
                                'PyTorch_ResNet18',
                                'PyTorch_ResNet34',
                                'PyTorch_ResNet50',
                                'PyTorch_ResNet101',
                                'PyTorch_ResNet152',
                                ], default_value='TensorFlow_ResNet9', key='-TRAIN DROPDOWN-',
                               background_color='#e3e3e3',
                               auto_size_text=True, expand_x=True, readonly=True, text_color='black',
                               enable_events=True)],
                  [sg.Frame('Optimizer settings:',
                            [[sg.Radio("SGD", group_id=3, default=True, key="-SGD-", enable_events=True,
                                       circle_color='blue'),
                              sg.Radio("Adam", group_id=3, key="-ADAM-", enable_events=True,
                                       circle_color='blue')]],
                            border_width=0, )],
                  [sg.Frame('Optimizer Learning rate:',
                            [[sg.Slider((0.001, 0.3), orientation='horizontal', resolution=0.001, pad=((0, 0), (0, 5)),
                                        default_value=0.001, relief=sg.RELIEF_FLAT, trough_color='#e3e3e3',
                                        key="-LR-",
                                        size=(25, 16))]],
                            border_width=0)],
                  [sg.Frame('Optimizer Decay:',
                            [[sg.Slider((0, 0.1), orientation='horizontal', resolution=0.001, pad=((0, 0), (0, 5)),
                                        default_value=0, relief=sg.RELIEF_FLAT, trough_color='#e3e3e3', key="-DECAY-",
                                        size=(25, 16))]],
                            border_width=0)],
                  [sg.Frame('Epochs:',
                            [[sg.Slider((1, 300), orientation='horizontal', resolution=1, pad=((0, 0), (0, 5)),
                                        default_value=30, relief=sg.RELIEF_FLAT, trough_color='#e3e3e3', key="-EPOCHS-",
                                        size=(25, 16))]],
                            border_width=0)],
                  [sg.Frame('Model save name:',
                            [[sg.Input(default_text='my_model',
                                       size=(25, 16), pad=((0, 0), (0, 5)), enable_events=True,
                                       key="-MODEL SAVE NAME-", )]],
                            border_width=0)],
              ], expand_x=True, expand_y=True, border_width=0, font=('Courier New', 11), ),
               sg.Frame('Train Dataset', [
                   [sg.Text('Choose train dataset\n(Folder with 7 subfolders named after the 7 emotions):')],
                   [sg.Text("Folder"),
                    sg.In(size=(35, 1), enable_events=True, key="-TRAIN FOLDER-", readonly=True, focus=False,
                          disabled_readonly_background_color='white'),
                    sg.FolderBrowse(),
                    ],
                   [sg.Frame('Files in subfolders:',
                             [[sg.Text(
                                 "anger:\n\ndisgust:\n\nfear:\n\nhappiness:\n\nneutrality:\n\nsadness:\n\nsurprise:\n",
                                 key="-TRAIN SUBFOLDERS-",
                                 background_color='white',
                                 auto_size_text=True,
                                 expand_y=True,
                                 expand_x=True,
                                 size=(52, 15)
                                 )]],
                             border_width=0, pad=((0, 0), (20, 0)))],
               ], expand_x=True, expand_y=True, border_width=0, pad=((25, 0), (0, 0)), font=('Courier New', 11)),

               ],

              [sg.Frame("",
                        [[sg.Button("Back", enable_events=True, size=(10, 1), font=('Courier New', 12)),
                          sg.Button("Train", enable_events=True, size=(10, 1), font=('Courier New', 12),
                                    disabled=True)]],
                        element_justification='center', border_width=0, pad=((0, 0), (60, 0)),
                        vertical_alignment='center')],
              ]
    return layout


def train_settings_loop(window, models):
    all_emotions_are_there = False
    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if "Back" in event:
            back_event(window)
            return models

        if event == "-TRAIN FOLDER-":
            folder = values["-TRAIN FOLDER-"]
            emotions = ["anger", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]

            trainfolderstr = "anger:\n\ndisgust:\n\nfear:\n\nhappiness:\n\nneutrality:\n\nsadness:\n\nsurprise:\n"
            window['-TRAIN SUBFOLDERS-'].update(trainfolderstr)
            all_emotions_are_there = True

            trainfolderstr = ""
            for emotion in emotions:
                if not os.path.isdir(os.path.join(folder, emotion)):
                    sg.PopupError(f"No '{emotion}' subfolder found in the chosen folder!", title='ERROR')
                    window["-TRAIN FOLDER-"].update('')
                    window['Train'].update(disabled=True)
                    all_emotions_are_there = False
                    break

                try:
                    # Get list of files in folder
                    file_list = os.listdir(os.path.join(folder, emotion))
                except:
                    file_list = []

                files = [f for f in file_list
                         if os.path.isfile(f"{folder}/{emotion}/{f}")
                         and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                imagescnt = len(files)
                if imagescnt == 0:
                    sg.PopupError(f"No images found in subfolder '{emotion}'!", title='ERROR')
                    window["-TRAIN FOLDER-"].update('')
                    window['Train'].update(disabled=True)
                    all_emotions_are_there = False
                    break

                trainfolderstr += f"{emotion}:\n {imagescnt} images found\n"

            if all_emotions_are_there:
                for f in os.listdir(folder):
                    if os.path.isdir(os.path.join(folder, f)):
                        if f not in ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']:
                            sg.PopupError(f"Bad subfolder in main folder: '{f}'", title='ERROR')
                            window["-TRAIN FOLDER-"].update('')
                            window['Train'].update(disabled=True)
                            all_emotions_are_there = False
                            break

            if all_emotions_are_there:
                window['-TRAIN SUBFOLDERS-'].update(trainfolderstr)
                window['Train'].update(disabled=False)

        if event == "-MODEL SAVE NAME-":
            chars = set(' /<>"\\\\|?*')
            name = values["-MODEL SAVE NAME-"]
            if any((c in chars) for c in name):
                sg.PopupError(f'Forbidden characters in name: /<>"\\|?*', title='ERROR')
                for c in chars:
                    name = name.replace(c, '')
                window["-MODEL SAVE NAME-"].update(name)
            if name == "":
                window['Train'].update(disabled=True)
            else:
                if all_emotions_are_there:
                    window['Train'].update(disabled=False)

        if event == "Train":
            window[f'-COL7-'].update(visible=False)
            window[f'-COL8-'].update(visible=True)
            models = train_loop(window, models)

            # window["-FILE LIST-"].update(fnames)
            # event = "-FILTER-"
            # if len(fnames) == 0:
            #     sg.PopupOK('No images found in folder!', title='SORRY')


if __name__ == "__main__":
    layout = train_settings_layout()
    window = sg.Window("Train Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
