import tensorflow as tf
import numpy as np
import random
import cv2
import os
import PySimpleGUI as sg

class tf_flags_StopSave():
    def __init__(self, stopped=False, save=False):
        self.stopped = stopped
        self.save = save

class StopTrainingOnWindowCloseAndPause(tf.keras.callbacks.Callback):
    """ NewCallback descends from Callback
    """
    def __init__(self, window, tf_flags):
        """ Save params in constructor
        """
        self.window = window
        self.tf_flags = tf_flags

    def on_train_batch_end(self, batch, logs=None):
        event, values = self.window.read(0)
        if event == "Exit" or event == sg.WIN_CLOSED or event == '-CANCEL_B-' or event == '-SAVE-':
            self.model.stop_training = True
            if event == '-CANCEL_B-':
                self.tf_flags.stopped = True
            if event == '-SAVE-':
                self.tf_flags.stopped = True
                self.tf_flags.save = True

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)

def conv_block_r9(in_channels, out_channels, pool=False):
    inputs = tf.keras.Input((None, None, in_channels))
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same')(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    if pool: results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def ResNet9(**kwargs):
    inputs = tf.keras.Input((None, None, 3))
    results =  conv_block_r9(in_channels=3, out_channels=64)(inputs)

    results =  conv_block_r9(64, 64, pool=True)(results)

    shortcut = conv_block_r9(64, 64, pool=True)(results)
    results = conv_block_r9(64, 32)(shortcut)
    results = conv_block_r9(32, 64)(results)
    results = tf.keras.layers.Add()([results, shortcut])
    results = tf.keras.layers.Dropout(0.5)(results)

    shortcut = conv_block_r9(64, 64, pool=True)(results)
    results = conv_block_r9(64, 32)(shortcut)
    results = conv_block_r9(32, 64)(results)
    results = tf.keras.layers.Add()([results, shortcut])
    results = tf.keras.layers.Dropout(0.5)(results)

    shortcut = conv_block_r9(64, 64, pool=True)(results)
    results = conv_block_r9(64, 32)(shortcut)
    results = conv_block_r9(32, 64)(results)
    results = tf.keras.layers.Add()([results, shortcut])
    results = tf.keras.layers.Dropout(0.5)(results)

    results = tf.keras.layers.MaxPool2D(pool_size=(6, 6))(results)
    results = tf.keras.layers.Flatten()(results)
    return tf.keras.Model(inputs = inputs, outputs = results, **kwargs)

def EmotionsRN9():
    inputs = tf.keras.Input((197, 197, 3))
    results = ResNet9(name = 'resnet9')(inputs)
    results = tf.keras.layers.Dense(7, activation = tf.keras.activations.softmax)(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def ResnetBlock(in_channels, out_channels, down_sample = False):
    inputs = tf.keras.Input((None, None, in_channels)) # inputs.shape = (batch, height, width, in_channels)
    if down_sample:
        shortcut = tf.keras.layers.Conv2D(out_channels, kernel_size = (1,1), strides = (2,2), padding = 'same')(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), strides = (2,2) if down_sample else (1,1), padding = 'same')(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), strides = (1,1), padding = 'same')(results)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.Add()([results, shortcut])
    results = tf.keras.layers.ReLU()(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def ResNet18(**kwargs):
    inputs = tf.keras.Input((None, None, 3))
    results = tf.keras.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same')(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 128, down_sample = True)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 256, down_sample = True)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 512, down_sample = True)(results)
    results = ResnetBlock(512, 512)(results)
    results = tf.keras.layers.GlobalAveragePooling2D()(results) # results.shape = (batch, 512)
    return tf.keras.Model(inputs = inputs, outputs = results, **kwargs)

def EmotionsRN18():
    inputs = tf.keras.Input((197, 197, 3))
    results = ResNet18(name = 'resnet18')(inputs)
    results = tf.keras.layers.Dense(7, activation = tf.keras.activations.softmax)(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def ResNet34(**kwargs):
    inputs = tf.keras.Input((None, None, 3))
    results = tf.keras.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same')(inputs)
    results = tf.keras.layers.BatchNormalization()(results)
    results = tf.keras.layers.ReLU()(results)
    results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 64)(results)
    results = ResnetBlock(64, 128, down_sample = True)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 128)(results)
    results = ResnetBlock(128, 256, down_sample = True)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 256)(results)
    results = ResnetBlock(256, 512, down_sample = True)(results)
    results = ResnetBlock(512, 512)(results)
    results = ResnetBlock(512, 512)(results)
    results = tf.keras.layers.GlobalAveragePooling2D()(results) # results.shape = (batch, 512)
    return tf.keras.Model(inputs = inputs, outputs = results, **kwargs)

def EmotionsRN34():
    inputs = tf.keras.Input((197, 197, 3))
    results = ResNet34(name = 'resnet34')(inputs)
    results = tf.keras.layers.Dense(7, activation = tf.keras.activations.softmax)(results)
    return tf.keras.Model(inputs = inputs, outputs = results)


# def facial_landmarks(image, predictor):
#     # image = cv2.imread(filepath)
#     face_rects = [dlib.rectangle(left=1, top=1, right=len(image) - 1, bottom=len(image) - 1)]
#     face_landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face_rects[0]).parts()])
#     return face_landmarks

# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def load_filenames(directory):
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
    samples = []
    for emotion in emotions_dict:
        path = directory + "/" + emotion
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                filepath = path + "/" + file
                emotion_label = emotions_dict[emotion]
                samples.append([filepath, emotion_label])
    return samples


def rotate_image(image, deg):
    rows, cols, c = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


def generator(samples, aug=False, batch_size=32, shuffle_data=True, resize=197, window=None):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        random.shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            X1 = []
            # X2 = []
            y = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_path = batch_sample[0]
                label = batch_sample[1]
                img = cv2.imread(img_path)
                img = cv2.resize(img, (resize, resize))
                if aug:  # augumentations
                    img = rotate_image(img, random.uniform(-10, 10))
                # features = facial_landmarks(img, predictor)
                img = img / 255

                onehot = [0 for i in range(7)]
                onehot[label] += 1

                # apply any kind of preprocessing
                # Add example to arrays
                X1.append(img)
                # X2.append(features)
                y.append(onehot)

            # Make sure they're numpy arrays (as opposed to lists)
            X1 = np.array(X1)
            # X2 = np.array(X2)
            y = np.array(y)

            if window:
                print('', end='')
                window.refresh()

            # The generator-y part: yield the next training batch
            # yield [X1, X2], y
            yield X1, y

def image_generator(dataset, aug=False, BS=32, get_datagen=False):
    if aug:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator.ImageDataGenerator(rescale=1./255)

    if get_datagen:
        return datagen
    return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197),
            color_mode='rgb',
            shuffle = True,
            class_mode='categorical',
            batch_size=BS)

if __name__ == "__main__":
    SGD_LEARNING_RATE = 0.01
    ADAM_LEARNING_RATE = 0.001
    SGD_DECAY = 0.0001
    EPOCHS = 5
    BS = 32
    Resize_pixelsize = 197
    sgd = tf.keras.optimizers.SGD(lr=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True)
    optim = tf.keras.optimizers.Adam(lr=ADAM_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    print('loading models')
    rn9 = EmotionsRN9()
    print(rn9)
    # rn18 = EmotionsRN18()
    # print(rn18)
    # rn34 = EmotionsRN34()
    # print(rn34)
    # rn50 = tf.keras.applications.ResNet50(weights=None, classes=7)
    # rn101 = tf.keras.applications.ResNet101(weights=None, classes=7)
    # rn152 = tf.keras.applications.ResNet152(weights=None, classes=7)
    # print(rn50)
    # rn34.save('rn34.h5')

    rn9.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',
                                                                           tf.keras.metrics.Precision(name='precision'),
                                                                           tf.keras.metrics.Recall(name='recall'),
                                                                           F1_Score(name='f1_score'),
                                                                           tf.keras.metrics.AUC(name='auc_roc')])

    samples_train = load_filenames("C:/Users/hp/Documents/data/train")
    # samples_dev = load_filenames("C:/Users/hp/Documents/data/dev")
    # samples_test = load_filenames("C:/Users/hp/Documents/data/test")
    #
    # train_generator = generator(samples_train, True)
    # dev_generator = generator(samples_dev)
    # test_generator = generator(samples_test)

    train_generator = image_generator("C:/Users/hp/Documents/data/train", True)

    metrics = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}


    # for ep in range(EPOCHS):
    #     history = rn9.fit_generator(
    #         generator=train_generator,
    #         # validation_data=dev_generator,
    #         steps_per_epoch=len(samples_train) // BS //100,
    #         # validation_steps=len(samples_dev) // BS,
    #         epochs=1,
    #         # use_multiprocessing=True
    #     )
    #     for key in metrics.keys():
    #         metrics[key].extend(history.history[key])
    #     print(metrics)
