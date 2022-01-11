import PySimpleGUI as sg
from settings_page import settings_layout, settings_loop
from base_page import base_layout
from load_page import load_layout, load_loop
from prediction_page import prediction_layout, predict_loop
from progress_page import progress_layout
from result_page import result_layout
from train_settings_page import train_settings_layout, train_settings_loop
from train_page import train_layout
from utils import back_event
import cv2
import os


class HECKApp:
    def __init__(self, *args, **kwargs):
        sg.theme("DefaultNoMoreNagging")
        sg.theme_button_color(('blue', '#b8cde0'))
        sg.set_options(font=('Courier New', 10))

        # ----------- Create all the layouts this Window will display -----------
        layout1 = base_layout()
        layout2 = load_layout()
        layout3 = settings_layout()
        layout4 = prediction_layout()
        layout5 = progress_layout()
        layout6 = result_layout()
        layout7 = train_settings_layout()
        layout8 = train_layout()

        # ----------- Create actual layout using Columns and a row of Buttons -----------
        layout = [[sg.Column(layout1, key='-COL1-', element_justification='center', vertical_alignment='c',
                             expand_y=True),
                   sg.Column(layout2, visible=False, key='-COL2-', element_justification='center', pad=(0, 0)),
                   sg.Column(layout3, visible=False, key='-COL3-', element_justification='center'),
                   sg.Column(layout4, visible=False, key='-COL4-', element_justification='center'),
                   sg.Column(layout5, visible=False, key='-COL5-', element_justification='center'),
                   sg.Column(layout6, visible=False, key='-COL6-', element_justification='center'),
                   sg.Column(layout7, visible=False, key='-COL7-', element_justification='center'),
                   sg.Column(layout8, visible=False, key='-COL8-', element_justification='center'),
                   ]]

        window = sg.Window('Human Emotion Classification Kit', layout, element_justification='center',
                           size=(800, 600))

        loaded_stuff = []
        faceCascade = cv2.CascadeClassifier('faceutils/haarcascade_frontalface_default.xml')
        models = {'res9pt': [], 'res50tf': []}
        predictor = None

        usr_model_path = os.getcwd().replace('\\', '/') + "/user_models"
        score_plots_path = os.getcwd().replace('\\', '/') + "/score_plots"
        if not os.path.isdir(usr_model_path):
            os.makedirs(usr_model_path)
        if not os.path.isdir(score_plots_path):
            os.makedirs(score_plots_path)

        while True:
            event, values = window.read()
            # print(event, values)
            # print(loaded_stuff)
            if event in (None, 'Exit'):
                break
            if event == 'Predict emotions':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL4-'].update(visible=True)
                models, predictor = predict_loop(window, loaded_stuff, faceCascade, models, predictor)
            if event == 'Load images':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL2-'].update(visible=True)
                loaded_stuff = load_loop(window, loaded_stuff)
            if event == 'Settings':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL3-'].update(visible=True)
                settings_loop(window, loaded_stuff, faceCascade)
            if event == 'Train models':
                window[f'-COL1-'].update(visible=False)
                window[f'-COL7-'].update(visible=True)
                models = train_settings_loop(window, models)
            if 'Back' in event:
                back_event(window)
        window.close()


if __name__ == "__main__":
    app = HECKApp()
