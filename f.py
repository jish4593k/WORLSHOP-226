import os
import shutil
import requests
import tempfile
from flask import Flask, render_template, request, send_file
from gevent.pywsgi import WSGIServer
from subprocess import call

# Additional Libraries
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'png', 'jpg', 'jpeg', 'gif'}

# AI Model (Example: Simple Logistic Regression)
def train_ai_model():
    # Load iris dataset as an example
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

class FileUploadForm(FlaskForm):
    file = FileField('Upload File')
    submit = SubmitField('Submit')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def api():
    work_dir = tempfile.TemporaryDirectory()
    input_file_path = os.path.join(work_dir.name, 'document')
    output_file_path = os.path.join(work_dir.name, 'document.pdf')
    # 
    ai_model = train_ai_model()

    form = FileUploadForm()

    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            file.save(input_file_path)

    convert_file(input_file_path, output_file_path)

    return render_template('index.html', form=form, ai_prediction=ai_prediction)

if __name__ == "__main__":
    app.run(debug=True)
