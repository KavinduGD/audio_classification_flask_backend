from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import librosa

model=load_model('./audio_classification_model_cnn.h5')
classes=['air_conditioner','car_horn','children_playing','dog_bark','drilling' , 'engine_idling', 'gun_shot','jackhammer', 'siren', 'street_music',]
def features_extractor(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>APP is running</h1>'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        features = features_extractor(file_path)
        # You can do something with the extracted features here
        result = model.predict(features.reshape(1,-1)).tolist()[0]  # Convert numpy array to list
        response = [{'class': cls, 'probability': prob} for cls, prob in zip(classes, result)]
        return jsonify({'result': response})
    else:
        return jsonify({'error': 'File type not allowed'})
    
if __name__ == '__main__':
    app.run()
