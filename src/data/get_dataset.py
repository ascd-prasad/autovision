import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import logging

project_path = '/Downloads/GL'


def load_train_test_data():
    pathToTrainData = 'Dataset/Car Images/Train Images'
    cars_train_data = load_data(pathToTrainData)
    logging.info("cars_train_data loaded and built successfully")
    cars_train_data.to_csv(r'references/cars_train_data.csv', index=False)
    logging.info("cars_train_data saved successfully")
    print(cars_train_data.head())
    print(cars_train_data.info())
    pathToTestData = 'Dataset/Car Images/Test Images'
    cars_test_data = load_data(pathToTestData)
    print(cars_test_data.head())
    print(cars_test_data.info())
    logging.info("cars_test_data loaded and built successfully")
    cars_test_data.to_csv(r'references/cars_test_data.csv', index=False)
    logging.info("cars_test_data saved successfully")
    cars_train_data.sort_values(['imageName'],axis=0,ascending=[True],inplace=True)
    cars_test_data.sort_values(['imageName'],axis=0,ascending=[True],inplace=True)
    logging.info("cars train and test data sorted successfully")
    logging.info('Renaming imageName to match to Annotations Data Set')
    cars_train_data.rename(columns = {'imageName': 'Image Name'},inplace = True)
    cars_test_data.rename(columns = {'imageName': 'Image Name'},inplace = True)
    print(cars_train_data.head())
    print(cars_test_data.head())
    return cars_train_data, cars_test_data


def load_data(pathToData):
    path = os.getcwd()
    print(path)
    # os.chdir(project_path)
    # print(os.getcwd())
    # Importing the data set
    data = pd.DataFrame(columns=['imageName', 'imagePath', 'class', 'height', 'width'])
    for dirname, _, filenames in os.walk(pathToData):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            img_name = os.path.split(path)[1]

            if img_name != '.DS_Store':
                img = cv2.imread(path)
                height, width, channel = img.shape
                class_label = dirname.split('/')[-1]
                data = data.append(
                    {'imageName': img_name, 'imagePath': path, 'class': class_label, 'height': height, 'width': width},
                    ignore_index=True)

    logging.info("Data loaded and built successfully")
    return data

def load_train_test_annotations():
    pathToAnotations ='Dataset/Annotations'
    cars_train_annotations = pd.read_csv(pathToAnotations+'/Train Annotations.csv')
    print(cars_train_annotations.head())
    print('Train anotations loaded')
    pathToAnotations ='Dataset/Annotations'
    cars_test_annotations = pd.read_csv(pathToAnotations+'/Test Annotation.csv')
    print(cars_test_annotations.head())
    print('Test anotations loaded')
    return cars_train_annotations,cars_test_annotations

def get_final_data(data, annotations):
    car_image_details = pd.merge(data, annotations,
                                       on='Image Name',
                                       how='outer')
    print(car_image_details.head())
    car_image_details.rename(columns = {'Bounding Box coordinates': 'X1','Unnamed: 2':'Y1','Unnamed: 3':'X2','Unnamed: 4':'Y2'},inplace = True)
    print(car_image_details.head())
    print(car_image_details['class'].value_counts())
    return car_image_details

