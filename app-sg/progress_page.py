import PySimpleGUI as sg
from utils import list_all_pictures, predict_res9pt, predict_res50tf, load_res9pt, load_res50tf


def progress_layout():
    BAR_MAX = 100
    layout = [[sg.Text('Please wait while prediction is in progress', font=('Courier New', 20), pad=((0, 0), (60, 0)))],
              [sg.HSep(pad=((0, 0), (0, 26)))],
              [sg.ProgressBar(max_value=BAR_MAX, orientation='h', size=(40, 40), key='-PROGRESS BAR-')],
              [sg.Button('Cancel', size=(10, 1), font=('Courier New', 12))]]
    return layout


def progress_loop(window, chosen_stuff, values, faceCascade, models):
    pic_list = list_all_pictures(chosen_stuff)
    num_pics = len(pic_list)
    steps = num_pics + 1
    # print('Counted:')
    # print(num_pics)

    test = []

    res9pt = values['-RESNET9-']
    res50tf = values['-RESNET50-']
    detection = values['-FACE DETECTION-']

    if res9pt and not models['res9pt']:
        models['res9pt'] = load_res9pt()
    elif res50tf and not models['res50tf']:
        models['res50tf'] = load_res50tf()

    i = 1
    window['-PROGRESS BAR-'].update(i, steps)

    for image_path in pic_list:
        event, values = window.read(0)

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if 'Cancel' in event:
            window[f'-COL5-'].update(visible=False)
            window[f'-COL4-'].update(visible=True)
            return models

        if res9pt:
            out = predict_res9pt(image_path, models['res9pt'], detection, faceCascade)
            test.append(out)

        # TODO
        elif res50tf:
            predict_res50tf(image_path, models['res50tf'], detection, faceCascade)

        i += 1
        window['-PROGRESS BAR-'].update(i, steps)

    print(test)


if __name__ == "__main__":
    layout = progress_layout()
    window = sg.Window("Progress Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
