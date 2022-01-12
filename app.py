from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import sys
import tensorflow as tf
import cv2
from PIL import Image
import pickle
import os

sys.path.append("..")

# from utils import visualization_utils as vis_util
print('current path:',os.getcwd())
saved_model_path = 'models/detection_model/saved_model'
model = tf.saved_model.load(saved_model_path)
label_class_dict = pickle.load(open('label_class_dict.pkl', 'rb'))

app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

#@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def login():
    return render_template("login.html")
database={'yafar':'123','vijay':'123','prasad':'123','Preetika':'123','jayshima':'123'}

@app.route('/form_login',methods=['POST','GET'])
def form_login():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
            return render_template('home.html',name=name1)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))

def detector_prediction(image_file, confidence_threshold=0.5):
    #Load image
    img = tf.keras.preprocessing.image.load_img(image_file)
    
    #Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')
    #Make it a batch of one example
    img_array = tf.expand_dims(img_array, axis=0)

    #Prediction
    output = model(img_array) #get list of tensors discussed above as output
    #print(output)
    detection_scores = output['detection_scores'].numpy()[0] #get detection scores
    detection_classes = output['detection_classes'].numpy()[0]
    detection_boxes = output['detection_boxes'].numpy()[0]
    #print(detection_scores)
    #Select predictions for which probability is higher than confidence_threshold
    selected_predictions = detection_scores >= confidence_threshold
    #print(selected_predictions)
    selected_prediction_scores = detection_scores[selected_predictions]
    selected_prediction_classes = detection_classes[selected_predictions]
    selected_prediction_boxes = detection_boxes[selected_predictions]

    #De-normalize box co-ordinates (multiply x-coordinates by image width and y-coords by image height)
    img_w, img_h = img.size

    for i in range(selected_prediction_boxes.shape[0]):
        
        selected_prediction_boxes[i,0] *= img_h #ymin * img_w
        selected_prediction_boxes[i,1] *= img_w #xmin * img_h
        selected_prediction_boxes[i,2] *= img_h #ymax * img_w
        selected_prediction_boxes[i,3] *= img_w #xmax * img_h

    #Make all co-ordinates as integer
    selected_prediction_boxes= selected_prediction_boxes.astype(int)

    #Convert class indexes to actual class labels
    predicted_classes = []
    for i in range(selected_prediction_classes.shape[0]):
        predicted_classes.append(label_class_dict[int(selected_prediction_classes[i])])

    #Number of predictions
    selected_num_predictions = selected_prediction_boxes.shape[0]

    return {'Total Predictions': selected_num_predictions,
            'Scores': selected_prediction_scores, 
            'Classes': predicted_classes, 
            'Box coordinates': selected_prediction_boxes}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]

    for image_file in TEST_IMAGE_PATHS:
        
        #Call model prediction function above
        output = detector_prediction(image_file, confidence_threshold=0.5)

        #Read image
        img = cv2.imread(image_file)

        #Draw rectangle for predicted boxes, also add predicted classes
        for i in range(output['Box coordinates'].shape[0]):

            box = output['Box coordinates'][i]
            
            #Draw rectangle - (ymin, xmin, ymax, xmax)
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0,0,255), 2)
            
            #Add Label - Class name and confidence level
            label = output['Classes'][i] + ': ' + str(round(output['Scores'][i],2))
            img = cv2.putText(img, label, (box[1], box[0]-10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,120,130), 1,cv2.LINE_AA)
        
        #Conver BGR image to RGB to use with Matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(img)
        im.save('uploads/' + filename)

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
