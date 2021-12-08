import os.path

import PySimpleGUI as sg
from utils import list_all_pictures, load_res9pt, load_res50tf, prediction_combo
from result_page import result_loop
import dlib


def progress_layout():
    BAR_MAX = 100
    layout = [[sg.Text('Please wait while prediction is in progress', font=('Courier New', 20), pad=((0, 0), (60, 0)))],
              [sg.HSep(pad=((0, 0), (0, 26)))],
              [sg.ProgressBar(max_value=BAR_MAX, orientation='h', size=(40, 40), key='-PROGRESS BAR-')],
              # [sg.Listbox([], background_color='white', key='-PROGRESS TEXT-', highlight_background_color='white', highlight_text_color='black',
              #             font=('Courier New', 12), size=(70, 15), enable_events=False, pad=(0, 20),
              #             no_scrollbar=True, )],
              [sg.Multiline(key='-PROGRESS TEXT-', font=('Courier New', 10), size=(80, 18), enable_events=False,
                            pad=(0, 20),
                            write_only=True, reroute_cprint=True, disabled=True)],
              [sg.Button('Cancel', size=(10, 1), font=('Courier New', 12), pad=(10, 20), key='-CANCEL-'),
               sg.Button('Results', size=(10, 1), font=('Courier New', 12), pad=(10, 20), disabled=True,
                         key='-CONTINUE-')]]
    return layout


def progress_loop(window, chosen_stuff, values, faceCascade, models, predictor):
    window['-CONTINUE-'].update(disabled=True)
    window['-CANCEL-'].update(text='Cancel')

    go_menu = False

    pic_list = list_all_pictures(chosen_stuff)
    num_pics = len(pic_list)
    # print('Counted:')
    # print(num_pics)
    result_dict = dict()
    change_dict = dict()

    res9pt = values['-RESNET9-']
    res50tf = values['-RESNET50-']
    detection = values['-FACE DETECTION-']
    save_dir = values['-RESULT FOLDER-']

    if res9pt:
        model_text = '-RESNET9-'
    else:
        model_text = '-RESNET50-'

    steps = num_pics + 2
    sg.cprint("* Starting prediction")
    i = 1

    # progress_text = ['Loading model...']
    # window['-PROGRESS TEXT-'].update(progress_text)
    window['-PROGRESS BAR-'].update(i, steps)

    if res9pt and not models['res9pt']:
        sg.cprint('* Loading model...')
        models['res9pt'] = load_res9pt()
    elif res50tf and not models['res50tf']:
        sg.cprint('Loading model...')
        models['res50tf'] = load_res50tf()
    else:
        sg.cprint('* Model already loaded')
    if not predictor:
        predictor = dlib.shape_predictor('faceutils/shape_predictor_68_face_landmarks.dat')

    i = 2
    window['-PROGRESS BAR-'].update(i, steps)
    # progress_text.append('Predicting emotions and saving results...')
    # window['-PROGRESS TEXT-'].update(progress_text)
    sg.cprint('* Predicting emotions and saving results...')

    saved_stuff = []

    for chosen_path in chosen_stuff:
        event, values = window.read(0)

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if '-CANCEL-' in event:
            sg.cprint("* Prediction cancelled", text_color='red')
            window[f'-COL5-'].update(visible=False)
            window[f'-COL4-'].update(visible=True)
            return models, predictor

        if os.path.isdir(chosen_path):
            pics_in_folder = list_all_pictures([chosen_path])
            text_folder = '* Predicting from: ' + chosen_path
            sg.cprint(text_folder)
            _, folder_name = os.path.split(chosen_path)
            temp_save_dir = f'{save_dir}/{folder_name}'
            saved_stuff.append(temp_save_dir)

            for image_path in pics_in_folder:
                try:
                    if res9pt:
                        # out = predict_res9pt(image_path, models['res9pt'])
                        result_tmp, change_tmp = prediction_combo(image_path, temp_save_dir, models['res9pt'], model_text, detection,
                                               faceCascade,
                                               values['-FD1-'], values['-FD2-'], values['-FD3-'])
                        result_dict.update(result_tmp)
                        change_dict.update(change_tmp)

                    elif res50tf:
                        # out = predict_res50tf(image_path, models['res50tf'], predictor)
                        result_tmp, change_tmp = prediction_combo(image_path, temp_save_dir, models['res50tf'], model_text, detection,
                                               faceCascade,
                                               values['-FD1-'], values['-FD2-'], values['-FD3-'], predictor)
                        result_dict.update(result_tmp)
                        change_dict.update(change_tmp)
                    tmp_text = '* Processed: ' + image_path
                    sg.cprint(tmp_text)
                    i += 1
                    window['-PROGRESS BAR-'].update(i, steps)
                except Exception:
                    sg.cprint('* ERROR PROCESSING: \'' + image_path + '\', IMAGE SKIPPED', text_color='red')
                    i += 1
                    window['-PROGRESS BAR-'].update(i, steps)

        else:
            try:
                if res9pt:
                    # out = predict_res9pt(image_path, models['res9pt'])
                    result_tmp, change_tmp = prediction_combo(chosen_path, save_dir, models['res9pt'], model_text, detection,
                                           faceCascade,
                                           values['-FD1-'], values['-FD2-'], values['-FD3-'])
                    result_dict.update(result_tmp)
                    change_dict.update(change_tmp)

                elif res50tf:
                    # out = predict_res50tf(image_path, models['res50tf'], predictor)
                    result_tmp, change_tmp = prediction_combo(chosen_path, save_dir, models['res50tf'], model_text, detection,
                                           faceCascade,
                                           values['-FD1-'], values['-FD2-'], values['-FD3-'], predictor)
                    result_dict.update(result_tmp)
                    change_dict.update(change_tmp)

                _, image_name = os.path.split(chosen_path)
                saved_path = f'{save_dir}/{image_name}'
                saved_stuff.append(saved_path)

                tmp_text = '* Processed: ' + chosen_path
                sg.cprint(tmp_text)
                i += 1
                window['-PROGRESS BAR-'].update(i, steps)
            except Exception:
                sg.cprint('* ERROR PROCESSING: \'' + chosen_path + '\', IMAGE SKIPPED', text_color='red')
                i += 1
                window['-PROGRESS BAR-'].update(i, steps)

    sg.cprint('* Done!')
    # progress_text.append('Done!')
    # window['-PROGRESS TEXT-'].update(progress_text)

    window['-CONTINUE-'].update(disabled=False)
    window['-CANCEL-'].update(text='Back')

    while True:
        if go_menu:
            return models, predictor, go_menu
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            break

        if '-CANCEL-' in event:
            window[f'-COL5-'].update(visible=False)
            window[f'-COL4-'].update(visible=True)
            return models, predictor, go_menu

        if event == '-CONTINUE-':
            window[f'-COL5-'].update(visible=False)
            window[f'-COL6-'].update(visible=True)
            go_menu = result_loop(window, saved_stuff, result_dict, change_dict)

    return models, predictor, go_menu


if __name__ == "__main__":
    layout = progress_layout()
    window = sg.Window("Progress Page", layout, element_justification='center',
                       size=(800, 600))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
