import PySimpleGUI as sg


def ColumnFixedSize(layout, size=(None, None), *args, **kwargs):
    return sg.Column([[sg.Column(
        [[sg.Sizer(0, size[1] - 1), sg.Column([[sg.Sizer(size[0] - 2, 0)]] + layout, *args, **kwargs, pad=(0, 0))]],
        *args, **kwargs)]], pad=(0, 0))


col_interior1 = [[sg.Text('My Window')],
                [sg.In()],
                [sg.In()],
                [sg.Button('Go'), sg.Button('Exit'), sg.Cancel(), sg.Ok()]]

layout1 = [[sg.Text('Below is a column that is 500 x 300')],
          [sg.Text('With the interior centered')],
          [ColumnFixedSize(col_interior1, size=(800, 600), background_color='grey', element_justification='c',
                           vertical_alignment='t')]]

def BackEvent(window):
    window[f'-COL2-'].update(visible=False)
    window[f'-COL3-'].update(visible=False)
    window[f'-COL4-'].update(visible=False)
    window[f'-COL1-'].update(visible=True)
