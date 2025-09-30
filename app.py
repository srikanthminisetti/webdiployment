from flask import Flask, render_template, request, redirect, url_for,session, send_from_directory,flash
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
import numpy as np
import os
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from numpy import loadtxt
from PIL import Image, ImageChops, ImageEnhance
import itertools
from sqlalchemy.orm import backref
from io import BytesIO
import re
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from keras.models import load_model
import joblib
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
import os



from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



#################################################################################################

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/treat')
def treat():
    return render_template('treatment.html')


#####################################################################################################

import numpy as np
import cv2
import joblib

model = joblib.load('models/model.joblib')

# Function to preprocess a single image
def preprocess_image(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))  # Resize the image to the required size
    return image.reshape(1, -1)  # Flatten the image to match the input shape

# Function to predict and return the result
def predict(image_path, model, size):
    # Load and preprocess the image
    preprocessed_image = preprocess_image(image_path, size)
    
    # Perform prediction using the trained model
    confidence_scores = model.predict_proba(preprocessed_image)[0]
    predicted_class = model.predict(preprocessed_image)[0]
    
    # Define class labels
    class_labels = {0: 'benign', 1: 'malignant', 2: 'normal'}
    
    # Get the predicted label and confidence score
    predicted_label = class_labels[predicted_class]
    confidence_score = confidence_scores[predicted_class]
    
    return predicted_label, confidence_score

###########################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Load the U-Net model
modelunet = load_model('models/unet_model.h5',compile=False)  # Load your trained U-Net model here

def load_image(path, size):
    image = cv2.imread(path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

def calculate_severity(predicted_mask,la):
    # Calculate the area of the white region in the predicted mask
    if la=='normal':
        return 0
    white_area = np.sum(predicted_mask)

    # Normalize the white area to the range [0, 1]
    normalized_white_area = white_area / (predicted_mask.shape[0] * predicted_mask.shape[1])

    # Severity is inversely proportional to the normalized white area
    severity = normalized_white_area*100

    return severity

def predict_and_display_single_image(image_path,la):
    # Load and preprocess the image
    image = load_image(image_path, size=128)

    # Perform prediction using the loaded model
    prediction = modelunet.predict(np.expand_dims(image, axis=0))

    # Calculate severity based on the predicted mask
    severity_percentage = calculate_severity(prediction[0][:, :, 0],la)

    # Save the predicted image with a unique filename
    predicted_image_path = f'static/testresults/predicted_image_{os.path.basename(image_path)}.png'
    plt.imsave(predicted_image_path, prediction[0][:, :, 0], cmap='gray')

    return predicted_image_path, severity_percentage




# Function to send email with image and prediction
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

# Function to send email with image and prediction
def send_prediction_email(email,un, image_data, class_name, confidence):
    print(session)
    # Construct the email message
    msg = MIMEMultipart()
    msg['Subject'] = 'Breast Cancer Prediction Result'
    msg['From'] = 'karthikreddi018@gmail.com'  # Replace with your email address
    msg['To'] = email

    # HTML content for the email body
    html_content = f"""
    <html>
        <body>
            <p style="font-size: 16px;">Hi,{un}</p>
            <p style="font-size: 16px;">Your prediction result</p>
            <p style="font-size: 16px;">Prediction: {class_name}</p>
            <p style="font-size: 16px;">Confidence: {confidence}%</p>
            <br>
            <p style="font-size: 16px; font-style: italic; color: #808080;">Thanks,</p>
            <p style="font-size: 16px; font-weight: bold; color: #808080;">BC-Team</p>
        </body>
    </html>
    """

    # Attach the HTML content to the email
    msg.attach(MIMEText(html_content, 'html'))

    # Attach the image to the email
    image = MIMEImage(image_data)
    image.add_header('Content-Disposition', 'attachment', filename='prediction_image.jpg')
    msg.attach(image)

    # Connect to Gmail's SMTP server
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('karthikreddi018@gmail.com', 'hzswgrtonjfnkhgf')  # Replace with your password
        smtp.sendmail(msg['From'], msg['To'], msg.as_string())




#############################################################################################

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        # lastname = request.form['lastname']
        email = request.form['email']
        # phone = request.form['phone']
        # gender = request.form['gender']
        # age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = 'static/uploads/'+filename
           
            

            # Make prediction and get the result
            predicted_label, confidence_score = predict(img, model, 128)
            la=predicted_label
            imgseg,sev=predict_and_display_single_image(img,la)
            apoint=round(sev,2)
            sevr = "N/A"  # Default value if severity is not available

            if sev is not None:  # Check if severity is available
                sev = int(sev)  # Convert severity to integer
                
                if sev < 5:
                    sevr = "Medium"
                elif 5 <= sev < 20:
                    sevr = "High"
                elif sev >= 20:
                    sevr = "Severe"
            

            return render_template('result.html', filename=filename, fn=firstname,  r=predicted_label,seg=imgseg,probab=confidence_score,s=apoint,c=sevr)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)






if __name__ == '__main__':
    #db.create_all()
    app.run(debug=True)
