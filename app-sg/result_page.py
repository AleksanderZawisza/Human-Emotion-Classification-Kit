import PySimpleGUI as sg
from PIL import Image
from io import BytesIO


def result_layout():
    file_list_result = [[sg.Frame('Result files folder',
                                  [[sg.Listbox(values=[], enable_events=True, size=(42, 21), key="-FILE LIST FINAL-",
                                               horizontal_scroll=True, highlight_background_color='#81b2db')]],
                                  border_width=0),
                         ]]

    image_result_column = [[sg.Frame('Image preview', [[sg.Image(key="-IMAGE RESULT-")]],
                                     size=(400, 275), border_width=0, pad=(0, 0),
                                     element_justification='center')],
                           ]

    col1 = sg.Column(file_list_result, size=(370, 475), element_justification='center')
    col2 = sg.Column(image_result_column, pad=(20, 3), vertical_alignment='top', element_justification='center')

    # ----- Full layout -----
    layout = [[sg.Column([[sg.Text('Results', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 0)))]])],
              [sg.Frame("", [[col1,
                              # sg.VSep(),
                              col2]], pad=(0, 0), border_width=0)],
              [sg.Frame("", [[
                  sg.Button('Back', enable_events=True, size=(10, 1), font=('Courier New', 12)),
                  sg.Button('Exit', enable_events=True, size=(10, 1), font=('Courier New', 12))]],
                        element_justification='center', border_width=0, pad=(0, 0),
                        vertical_alignment='center')], ]

    return layout


def result_loop(window, saved_stuff):
    window['-FILE LIST FINAL-'].update(saved_stuff)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Back' in event:
            window[f'-COL6-'].update(visible=False)
            window[f'-COL4-'].update(visible=True)
            return

        if event == "-FILE LIST FINAL-":
            file_path = values["-FILE LIST FINAL-"][0]
            print(file_path)
            im = Image.open(file_path)
            width, height = (350, 250)
            scale = max(im.width / width, im.height / height)
            if scale > 1:
                w, h = int(im.width / scale), int(im.height / scale)
                im = im.resize((w, h), resample=Image.CUBIC)
            with BytesIO() as output:
                im.save(output, format="PNG")
                data = output.getvalue()
            window["-IMAGE RESULT-"].update(data=data)

    window.close()
