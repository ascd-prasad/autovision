import logging

import argparse

from src.data import get_dataset
from src.visualization import visualize

logging.getLogger().setLevel(logging.INFO)

#logger = log.setup_custom_logger('root')

#from src import data,visualization
#import data.get_dataset
# from visualization import visualize
from flask import Flask, jsonify
import threading

import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'up'}), 200

def run_api():
    hostname='localhost'
    portname='8080'
    app.run(host=hostname,port=portname)

@app.route('/autovision_model_training', methods=['GET'])
def run_autovision_model_training():
    logging.info("Model Training Process Started")

    cars_train_data, cars_test_data = get_dataset.load_train_test_data()
    logging.info("Visualizing Training images")
    visualize.visualize_images(cars_train_data)
    visualize.visualize_images(cars_test_data)

    cars_train_annotations,cars_test_annotations = get_dataset.load_train_test_annotations()
    car_train_image_details = get_dataset.get_final_data(cars_train_data,cars_train_annotations)
    car_test_image_details = get_dataset.get_final_data(cars_test_data,cars_test_annotations)
    visualize.visualize_df_images(car_train_image_details)
    visualize.visualize_df_images(car_test_image_details)
    visualize.display_image_with_bounding_box(car_train_image_details.iloc[10,:])
    visualize.display_image_with_bounding_box(car_train_image_details.iloc[3333,:])
    visualize.visualize_df_images_with_bounding_box(car_train_image_details)

    visualize.display_image_with_bounding_box(car_test_image_details.iloc[10,:])
    visualize.display_image_with_bounding_box(car_test_image_details.iloc[3333,:])
    visualize.visualize_df_images_with_bounding_box(car_test_image_details)
    logging.info("Completed autovision_model_training")
    return jsonify({'status': 'successfully executed'}), 200


def main():
    flask_thread=threading.Thread(target=run_api(),daemon=True)
    flask_thread.start()
    logging.info("Main Process Started")

main()