import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import os
from utils import list_all_pictures, create_result_text, create_result_text_folder


def result_layout():
    col_l = sg.Column([[sg.Frame('Prediction results', [[sg.Text(key="-PREDICTION RESULTS-",
                                                                 background_color='white',
                                                                 auto_size_text=True,
                                                                 expand_y=True,
                                                                 expand_x=True)]],
                                 border_width=0, pad=(0, 0), size=(250, 400),
                                 element_justification='center')]])

    col_r = sg.Column([[sg.Frame('Image preview', [[sg.Image(key="-IMAGE RESULT-")]],
                                 size=(550, 400), border_width=0, pad=(0, 0),
                                 element_justification='center')]])

    layout = [[sg.Frame('Choose image/folder from results:',
                        [[sg.DropDown([''], background_color='#e3e3e3', key='-FOLDERPIC DROPDOWN-',
                                      auto_size_text=True, expand_x=True, readonly=True, text_color='black',
                                      enable_events=True, )],
                         [sg.Text('Choose image from folder chosen above:')],
                         [sg.DropDown(values=[],
                                      background_color='#e3e3e3', key='-PIC DROPDOWN-',
                                      auto_size_text=True, expand_x=True, readonly=True, text_color='black',
                                      enable_events=True)]
                         ], size=(800, 100))],
              [sg.Frame("", [[col_l, col_r]], pad=(0, 0), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12)),
                  sg.Button('Main menu', key='-MENU-', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center')],
              ]
    return layout


def show_image_result(chosen_path, window):
    print(chosen_path)
    im = Image.open(chosen_path)
    width, height = (600, 400)
    scale = max(im.width / width, im.height / height)
    w, h = int(im.width / scale), int(im.height / scale)
    im = im.resize((w, h), resample=Image.CUBIC)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    window["-IMAGE RESULT-"].update(data=data)


def result_loop(window, saved_stuff, result_dict, change_dict):
    window['-FOLDERPIC DROPDOWN-'].update(values=saved_stuff)
    window['-FOLDERPIC DROPDOWN-'].expand()
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Back' in event:
            window[f'-COL6-'].update(visible=False)
            window[f'-COL5-'].update(visible=True)
            return False

        if event == '-MENU-':
            window[f'-COL6-'].update(visible=False)
            return True

        if event == '-FOLDERPIC DROPDOWN-':
            chosen_path = values['-FOLDERPIC DROPDOWN-']
            if os.path.isdir(chosen_path):
                pics = list_all_pictures([chosen_path])
                window['-PIC DROPDOWN-'].update(values=pics)
                result_text = create_result_text_folder(result_dict, chosen_path)
                result_text = 'Average in folder: \n' + result_text
                window['-PREDICTION RESULTS-'].update(result_text)
                window["-IMAGE RESULT-"].update(data=[])
            else:
                window['-PIC DROPDOWN-'].update(values=[])
                show_image_result(chosen_path, window)
                print(result_dict)
                result_text = create_result_text(result_dict[chosen_path])
                result_text = 'Average in photo: \n' + result_text
                window['-PREDICTION RESULTS-'].update(result_text)

        if event == '-PIC DROPDOWN-':
            chosen_pic = values['-PIC DROPDOWN-']
            show_image_result(chosen_pic, window)
            print(result_dict)
            result_text = create_result_text(result_dict[chosen_pic])
            result_text = 'Average in photo: \n' + result_text
            window['-PREDICTION RESULTS-'].update(result_text)

    window.close()
