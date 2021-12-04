import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Model, load_model
from sklearn import preprocessing
from tensorflow.keras.layers import Input, InputLayer, Dense, Dropout, BatchNormalization, Flatten, Activation, Conv2D, \
    MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, precision_score, \
    recall_score, f1_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing import image


def build_model(car_train_image_details):
    # Creating a Dictionary of training image classes.
    num_classes = car_train_image_details['Image class'].unique()
    label_class_dict = dict(zip(car_train_image_details['class'], car_train_image_details['Image class']))
    print(label_class_dict)
    image_size = 256
    target_size = (256, 256)
    batch_size = 100
    seed = 7
    nb_epoch = 100
    width = 256
    height = 256
    inception_model = define_model(width, height)
    print(inception_model.summary())
    pathToTrainData = 'Dataset/Car Images/Train Images'
    train_generator, validation_generator = define_generators(pathToTrainData)
    #train_generator = build_train_data(car_train_image_details, batch_size=batchsize)
    save_callback = define_callbacks()
    model_history = inception_model.fit(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data= validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=save_callback )
    print(model_history)
    print('Model trained and saved successfully')


def define_model(width, height):
    model_input = Input(shape=(width, height, 3), name='image_input')
    model_main = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')(
        model_input)
    model_dense1 = Flatten()(model_main)
    model_dense2 = Dense(256, activation='relu')(model_dense1)
    model_out = Dense(196, activation="softmax")(model_dense2)

    model = Model(model_input, model_out)
    optimizer = Adam(lr=0.00004, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def build_train_data(data, batch_size):
    index = np.random.randint(0, data.shape[0], size=batch_size)
    train_images = np.zeros(shape=(batch_size, width, height, 3))
    bounding_boxes = np.zeros(shape=(batch_size, 4))
    class_labels = np.zeros(shape=(batch_size, len(num_classes)))

    for i in range(batch_size):
        path = data.loc[index[i], 'imagePath']
        img = image.load_img(path, target_size=(width, height))
        img_array = image.img_to_array(img)
        train_images[i] = img_array

        cls_label = data.loc[index[i], 'Image class']
        cls_label = to_categorical(cls_label - 1, num_classes=len(num_classes), dtype='float32')
        class_labels[i] = cls_label

        # Normalizing bounding box co-ordinates
        image_width = data.loc[index[i], 'width']
        image_height = data.loc[index[i], 'height']

        x1 = data.loc[index[i], 'X1'] * image_size / image_width
        x2 = data.loc[index[i], 'X2'] * image_size / image_width
        y1 = data.loc[index[i], 'Y1'] * image_size / image_height
        y2 = data.loc[index[i], 'Y2'] * image_size / image_height
        bounding_boxes[i] = [x1, y1, x2 - x1, y2 - y1]

        # Normalizing batch images
        for i in range(batch_size):
            train_images[i] = train_images[i] / 255.0

        bounding_boxes = bounding_boxes / image_size

        return train_images, [class_labels, bounding_boxes]


def define_generators(pathToTrainData):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory=pathToTrainData,
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='training',
        classes=label_class_dict
    )

    validation_generator = train_datagen.flow_from_directory(
        directory=pathToTrainData,
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='validation',
        classes=label_class_dict
    )

    return train_generator, validation_generator

def define_callbacks():
    es = EarlyStopping(monitor='loss', patience=5, mode='min', min_delta=0.0001)
    checkpoint = ModelCheckpoint( filepath='/models/car_inception_model.h5',
                                  monitor='val_accuracy',
                                  save_best_only=True,
                                  verbose=1)
    save_callback = [checkpoint,es]

    return save_callback
