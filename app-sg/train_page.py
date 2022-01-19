import os
import time

import PySimpleGUI as sg
import numpy as np
import torch.optim
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

from utils_pt_train import *
from utils_tf_train import *

from keras import backend as K


def train_layout():
    layout = [[sg.Column([[sg.Text('Training in progress', font=('Courier New', 20))],
                          [sg.HSep(pad=((0, 0), (0, 0)))]])],
              [sg.Multiline(key='-PROGRESS TEXT TRAIN-', font=('Courier New', 10), size=(90, 8), enable_events=False,
                            pad=(0, 20), reroute_stdout=True, auto_refresh=True,
                            write_only=True, reroute_cprint=True, disabled=True)],
              [sg.Frame('Training scores:', [[sg.Image(key="-GRAPH-", background_color='white')]],
                        size=(550, 350), border_width=0, pad=(0, 0),
                        element_justification='center'),
               sg.Frame('', [
                   [sg.Button('Stop & Save', size=(15, 1), font=('Courier New', 12), pad=((10, 10), (90, 20)),
                              key='-SAVE-')],
                   [sg.Button('Cancel & Back', size=(15, 1), font=('Courier New', 12), pad=(10, 20), key='-CANCEL_B-')],
                   [sg.Button('Main menu', size=(15, 1), font=('Courier New', 12), pad=(10, 20), key='-MENU_B-',
                              disabled=True)],
               ],
                        size=(250, 350), border_width=0, pad=(0, 0), element_justification='center')],
              ]
    return layout


def train_epoch_pt(epoch, model, history, optimizer, train_loader, window, grad_clip=None):
    train_losses = []
    stopped = False
    save = False
    # predss = []
    # labelss = []
    accs = []
    f1s = []
    recalls = []
    precisions = []
    auc_rocs = []

    i = 1
    n = len(train_loader)
    for batch in train_loader:
        event, values = window.read(0)

        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            stopped = True
            save = False
            return history, stopped, save

        if event == '-CANCEL_B-':
            stopped = True
            save = False
            return history, stopped, save

        if event == '-SAVE-':
            stopped = True
            save = True
            return history, stopped, save

        data = model.training_step(batch)
        loss = data[0]
        preds = data[1].cpu().numpy()
        labels = data[2].cpu().numpy()
        acc_sc, f1_sc, recall_sc, precision_sc = all_scores(preds, labels)
        accs.append(acc_sc)
        f1s.append(f1_sc)
        recalls.append(recall_sc)
        precisions.append(precision_sc)
        try:
            ar_sc = auc_roc_sc(labels, preds)
        except:
            ar_sc = 0.5
        auc_rocs.append(ar_sc)

        sg.cprint("Batch {}/{} BATCH ACC: {:.2f}".format(i, n, acc_sc), key="-PROGRESS TEXT TRAIN-")
        i += 1
        # predss.extend(preds)
        # labelss.extend(labels)
        train_losses.append(loss)
        loss.backward()

        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()

    # Validation phase
    # print("Preds len " + str(len(predss)))
    # print("Labels len " + str(len(labelss)))
    result = {}
    result['train_loss'] = torch.stack(train_losses).mean().item()
    # acc_sc, f1_sc, recall_sc, precision_sc = all_scores(labelss, predss)
    result['train_acc'] = np.mean(accs)
    # print(all_scores(labelss, predss))
    result['train_f1'] = np.mean(f1s)
    print("train_f1: " + str(result['train_f1']))
    result['train_recall'] = np.mean(recalls)
    print("train_recall: " + str(result['train_recall']))
    result['train_auc_roc'] = np.mean(auc_rocs)
    print("train_auc_roc: " + str(result['train_auc_roc']))
    result['train_precision'] = np.mean(precisions)
    print("train_precision: " + str(result['train_precision']))
    model.epoch_end(epoch, result)
    history.append(result)
    return history, stopped, save


def save_scores_plot(history, model_name, n_epochs, epoch, model_savename, is_last=False):
    title = f"{model_savename} ({model_name}) model accuracy"

    if "PyTorch" in model_name:
        train_losses = [x['train_loss'] for x in history]
        train_accs = [x['train_acc'] for x in history]
        train_precisions = [x['train_precision'] for x in history]
        train_recalls = [x['train_recall'] for x in history]
        train_f1s = [x['train_f1'] for x in history]
        train_auc_rocs = [x['train_auc_roc'] for x in history]
    if "TensorFlow" in model_name:
        train_losses = history['loss']
        train_accs = history['acc']
        train_precisions = history['precision']
        train_recalls = history['recall']
        train_f1s = history['f1_score']
        train_auc_rocs = history['auc_roc']

    fig, ax1 = plt.subplots()
    ax1.set_ylim([0, max(train_losses)])

    ax1.plot(train_losses, color='crimson')
    plt.ylabel('loss')
    plt.legend(['crossentropy_loss'], loc='lower right')

    ax2 = ax1.twinx()

    ax2.plot(train_precisions, color='limegreen')
    ax2.plot(train_recalls, color='orange')
    ax2.plot(train_f1s, color='violet')
    ax2.plot(train_auc_rocs, color='blueviolet')
    ax2.plot(train_accs, color='dodgerblue')
    plt.ylabel('accuracy scores')
    plt.ylim([0, 1])

    plt.title(title)
    plt.xlabel('epoch')
    plt.xlim([0, n_epochs - 1])
    plt.legend(['precision', 'recall', 'f1_score', 'auc_roc', 'accuracy'], loc='upper left')
    file_name = model_savename + '_(' + model_name + ').png'
    if is_last:
        file_path = os.getcwd() + '/model_scores/' + file_name
    else:
        file_path = os.getcwd() + '/score_plots/' + file_name
    plt.savefig(file_path, bbox_inches='tight', transparent=True)
    return file_path


def train_loop(window, models):
    event, values = window.read(0)

    try:
        K.clear_session()
        torch.cuda.empty_cache()
        del models
        models = {}
    except:
        pass

    window["-CANCEL_B-"].update(text="Cancel & Back")
    window["-SAVE-"].update(text="Stop & Save")
    window["-MENU_B-"].update(disabled=True)
    window[f"-PROGRESS TEXT TRAIN-"].update("")
    window[f"-GRAPH-"].update("")
    go_menu_b = False

    n_epochs = int(values['-EPOCHS-'])
    lr = values['-LR-']
    weight_decay = values['-DECAY-']
    isAdam = values['-ADAM-']
    model_name = values['-TRAIN DROPDOWN-']
    model_savename = values["-MODEL SAVE NAME-"]
    data_dir = values['-TRAIN FOLDER-']

    sg.cprint("* Training has started", key="-PROGRESS TEXT TRAIN-")

    if 'PyTorch' in model_name:
        train_loader, device = make_train_loader_pt(data_dir, 64)
        if model_name == 'PyTorch_ResNet9':
            model = to_device(ResNet(1, 7), device)
        elif model_name == 'PyTorch_ResNet18':
            model = to_device(ResNet18_pt(1, 7), device)
        elif model_name == 'PyTorch_ResNet34':
            model = to_device(ResNet34_pt(1, 7), device)
        elif model_name == 'PyTorch_ResNet50':
            model = to_device(ResNet50_pt(1, 7), device)
        # elif model_name == 'PyTorch_ResNet101':
        #     model = to_device(ResNet101_pt(1, 7), device)
        # else:
        #     model = to_device(ResNet152_pt(1, 7), device)

        if isAdam:
            opt_func = torch.optim.Adam
        else:
            opt_func = torch.optim.SGD
        optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
        model.train()
    if 'TensorFlow' in model_name:
        if model_name == 'TensorFlow_ResNet9':
            model = EmotionsRN9()
        elif model_name == 'TensorFlow_ResNet18':
            model = EmotionsRN18()
        elif model_name == 'TensorFlow_ResNet34':
            model = EmotionsRN34()
        elif model_name == 'TensorFlow_ResNet50':
            model = tf.keras.applications.ResNet50(weights=None, classes=7, input_shape=(197, 197, 3))
        # elif model_name == 'TensorFlow_ResNet101':
        #     model = tf.keras.applications.ResNet101(weights=None, classes=7, input_shape=(197, 197, 3))
        # else:
        #     model = tf.keras.applications.ResNet152(weights=None, classes=7, input_shape=(197, 197, 3))

        if isAdam:
            optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=weight_decay, nesterov=True)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',
                                                                                     tf.keras.metrics.Precision(
                                                                                         name='precision'),
                                                                                     tf.keras.metrics.Recall(
                                                                                         name='recall'),
                                                                                     F1_Score(name='f1_score'),
                                                                                     tf.keras.metrics.AUC(
                                                                                         name='auc_roc')])
        BS = 16
        train_generator = image_generator(data_dir, True, BS=BS)
        samples_train = load_filenames(data_dir)
        # train_generator = generator(samples_train, True, batch_size=BS,  window=window)
        tf_metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}

    sg.cprint("* Model has been created", key="-PROGRESS TEXT TRAIN-")
    history = []
    tf_flags = tf_flags_StopSave()
    stopped = False
    save = False
    for epoch in range(n_epochs):
        event, values = window.read(0)

        if 'PyTorch' in model_name:
            sg.cprint(f'EPOCH [{epoch}]', end='\n', key="-PROGRESS TEXT TRAIN-")
            history, stopped, save = train_epoch_pt(epoch, model, history, optimizer, train_loader, window,
                                                    grad_clip=0.2)
            if len(history) > 0:
                filepath = save_scores_plot(history, model_name, n_epochs, epoch, model_savename, False)
        if 'TensorFlow' in model_name:
            sg.cprint(f'EPOCH [{epoch}]', end='', key="-PROGRESS TEXT TRAIN-")
            callback = StopTrainingOnWindowCloseAndPause(window, tf_flags)
            history = model.fit(train_generator, steps_per_epoch=len(samples_train) // BS, epochs=1,
                                callbacks=[callback])
            stopped = callback.tf_flags.stopped
            save = callback.tf_flags.save
            for key in tf_metrics.keys():
                tf_metrics[key].extend(history.history[key])

            filepath = save_scores_plot(tf_metrics, model_name, n_epochs, epoch, model_savename, False)

        event, values = window.read(0)
        if event == "Exit" or event == sg.WIN_CLOSED or event is None:
            return models, go_menu_b

        if event == '-CANCEL_B-':
            window[f'-COL8-'].update(visible=False)
            window[f'-COL7-'].update(visible=True)
            if "PyTorch" in model_name:
                try:
                    del model
                    del train_loader
                    torch.cuda.empty_cache()
                except:
                    pass
            elif "TensorFlow" in model_name:
                try:
                    del model
                    del train_generator
                    K.clear_session()
                except:
                    pass
            return models, go_menu_b

        if event == '-SAVE-':
            models[model_name] = model
            if "PyTorch" in model_name:
                temp = save_scores_plot(history, model_name, n_epochs, epoch, model_savename, True)
                model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.pth'
                torch.save(model.state_dict(), model_path)
                try:
                    del model
                    del train_loader
                    torch.cuda.empty_cache()
                except:
                    pass
            elif "TensorFlow" in model_name:
                temp = save_scores_plot(tf_metrics, model_name, n_epochs, epoch, model_savename, True)
                model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.h5'
                model.save_weights(model_path)
                try:
                    del model
                    del train_generator
                    K.clear_session()
                except:
                    pass
            sg.cprint("* Model has been saved", key="-PROGRESS TEXT TRAIN-", text_color='green')
            time.sleep(2)
            window[f'-COL8-'].update(visible=False)
            window[f'-COL7-'].update(visible=True)
            return models, go_menu_b

        if stopped:
            if save:
                if 'TensorFlow' in model_name:
                    sg.cprint("\n* Training was manually stopped", text_color='blue', key="-PROGRESS TEXT TRAIN-")
                else:
                    sg.cprint("* Training was manually stopped", text_color='blue', key="-PROGRESS TEXT TRAIN-")
                models[model_name] = model
                if "PyTorch" in model_name:
                    if len(history) > 0:
                        temp = save_scores_plot(history, model_name, n_epochs, epoch, model_savename, True)
                    model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.pth'
                    torch.save(model.state_dict(), model_path)
                    try:
                        del model
                        del train_loader
                        torch.cuda.empty_cache()
                    except:
                        pass
                elif "TensorFlow" in model_name:
                    temp = save_scores_plot(tf_metrics, model_name, n_epochs, epoch, model_savename, True)
                    model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.h5'
                    model.save_weights(model_path)
                    try:
                        del model
                        del train_generator
                        K.clear_session()
                    except:
                        pass

                sg.cprint("* Model has been saved", key="-PROGRESS TEXT TRAIN-", text_color='green')
                time.sleep(2)
                window[f'-COL8-'].update(visible=False)
                window[f'-COL7-'].update(visible=True)
                return models, go_menu_b
            else:
                if 'TensorFlow' in model_name:
                    sg.cprint("\n* Training cancelled", text_color='red', key="-PROGRESS TEXT TRAIN-")
                    try:
                        del model
                        del train_generator
                        K.clear_session()
                    except:
                        pass
                else:
                    sg.cprint("* Training cancelled", text_color='red', key="-PROGRESS TEXT TRAIN-")
                    try:
                        del model
                        del train_loader
                        torch.cuda.empty_cache()
                    except:
                        pass
                time.sleep(2)
                window[f'-COL8-'].update(visible=False)
                window[f'-COL7-'].update(visible=True)
                return models, go_menu_b
        else:
            pass

        try:
            im = Image.open(filepath)
        except:
            pass
        width, height = (520, 320)
        scale = max(im.width / width, im.height / height)
        if scale > 1:
            w, h = int(im.width / scale), int(im.height / scale)
            im = im.resize((w, h), resample=Image.CUBIC)
        with BytesIO() as output:
            im.save(output, format="PNG")
            data = output.getvalue()
        plt.close('all')
        window["-GRAPH-"].update(data=data)
        try:
            path = os.getcwd() + "/score_plots"
            for filename in os.listdir(path):
                file_path = path + "/" + filename
                os.remove(file_path)
        except:
            pass

    sg.cprint('* Training finished', key="-PROGRESS TEXT TRAIN-")

    window[f"-CANCEL_B-"].update(text="Go back")
    window[f"-SAVE-"].update(text="Save model")
    window[f"-MENU_B-"].update(disabled=False)

    while True:
        event, values = window.read(0)

        if event == "Exit" or event == sg.WIN_CLOSED:
            return models, go_menu_b

        if event == '-CANCEL_B-':
            if "PyTorch" in model_name:
                try:
                    del model
                    del train_loader
                    torch.cuda.empty_cache()
                except:
                    pass
            elif "TensorFlow" in model_name:
                try:
                    del model
                    del train_generator
                    K.clear_session()
                except:
                    pass
            window[f'-COL8-'].update(visible=False)
            window[f'-COL7-'].update(visible=True)
            return models, go_menu_b

        if event == '-SAVE-':
            models[model_name] = model
            if "PyTorch" in model_name:
                temp = save_scores_plot(history, model_name, n_epochs, epoch, model_savename, True)
                model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.pth'
                torch.save(model.state_dict(), model_path)
                try:
                    del model
                    del train_loader
                    torch.cuda.empty_cache()
                except:
                    pass
            elif "TensorFlow" in model_name:
                temp = save_scores_plot(tf_metrics, model_name, n_epochs, epoch, model_savename, True)
                model_path = os.getcwd() + "/user_models/" + model_savename + '_(' + model_name + ')' + '.h5'
                model.save_weights(model_path)
                try:
                    del model
                    del train_generator
                    K.clear_session()
                except:
                    pass

            sg.cprint("* Model has been saved", key="-PROGRESS TEXT TRAIN-", text_color='green')
            time.sleep(2)
            window[f'-COL8-'].update(visible=False)
            window[f'-COL7-'].update(visible=True)
            return models, go_menu_b

        if event == '-MENU_B-':
            if "PyTorch" in model_name:
                try:
                    del model
                    del train_loader
                    torch.cuda.empty_cache()
                except:
                    pass
            elif "TensorFlow" in model_name:
                try:
                    del model
                    del train_generator
                    K.clear_session()
                except:
                    pass
            window[f'-COL8-'].update(visible=False)
            window[f'-COL7-'].update(visible=True)
            go_menu_b = True
            return models, go_menu_b


if __name__ == "__main__":
    train_loader, device = make_train_loader_pt("C:/Users/Aleksander Podsiad/Desktop/Projekt Emocje/Dane/test", 64)
    # model = to_device(ResNet(1, 7), device)
    model = torch.load(
        'C:/Users/Aleksander Podsiad/Desktop/Human-Emotion-Classification-Kit/app-sg/models/ResNet1_mdl_EPOCHS_40.pth')
    model = to_device(model, device)
    # opt_func = torch.optim.Adam
    # model.train()
    # history = []
    # lr = 0
    # weight_decay = 0
    # optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    # history, stopped, save = train_epoch_pt(1, model, history, optimizer, train_loader, None,
    #                                         grad_clip=0.2)
    # print(history)

    # model.eval()
    # preds = []
    # labels = []
    # for batch in train_loader:
    #     pred, label = model.test_func(batch)
    #     preds.extend(pred.cpu())
    #     labels.extend(label.cpu())
    # labels = [x.item() for x in labels]
    # preds = [x.item() for x in preds]
    # print(accuracy_score(labels, preds))
    # print(recall_sc(labels, preds))
