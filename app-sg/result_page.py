import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import os


def result_layout():
    col_l = sg.Column([[sg.Frame('Prediction results', [[sg.Text(key="-PREDICTION RESULTS-")]],
                                 size=(400, 300), border_width=0, pad=(0, 0),
                                 element_justification='center')]])

    col_r = sg.Column([[sg.Frame('Image preview', [[sg.Image(key="-IMAGE RESULT-")]],
                                 size=(400, 300), border_width=0, pad=(0, 0),
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
                  sg.Button('Main menu', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center')],
              ]

    # ----- Full layout -----
    # layout = [[sg.Column([[sg.Text('Results', font=('Courier New', 20))],
    #                       [sg.HSep(pad=((0, 0), (0, 0)))]])],
    #
    #           [sg.Frame("", [[col1,
    #                           # sg.VSep(),
    #                           col2]], pad=(0, 0), border_width=0)],
    #           [sg.Frame("", [[
    #               sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12)),
    #               sg.Button('Exit', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
    #                     element_justification='center', border_width=0, pad=(0, 0),
    #                     vertical_alignment='center')], ]

    return layout


def show_image_result(chosen_path, window):
    im = Image.open(chosen_path)
    width, height = (400, 300)
    scale = max(im.width / width, im.height / height)
    if scale > 1:
        w, h = int(im.width / scale), int(im.height / scale)
        im = im.resize((w, h), resample=Image.CUBIC)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    window["-IMAGE RESULT-"].update(data=data)


def result_loop(window, saved_stuff):
    window['-FOLDERPIC DROPDOWN-'].update(values=saved_stuff)
    window['-FOLDERPIC DROPDOWN-'].expand()
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Back' in event:
            window[f'-COL6-'].update(visible=False)
            window[f'-COL4-'].update(visible=True)
            return

        if event == '-FOLDERPIC DROPDOWN-':
            chosen_path = values['-FOLDERPIC DROPDOWN-'][0]
            if os.path.isdir(chosen_path):
                pics = os.listdir(chosen_path)
                window['-PIC DROPDOWN-'].update(values=pics)
            else:
                show_image_result(chosen_path, window)

        if event == '-PIC DROPDOWN' and values['-PIC DROPDOWN']:
            chosen_pic = values['-FOLDERPIC DROPDOWN-'][0]
            show_image_result(chosen_pic, window)

    window.close()
