from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)
model = load_model('main.keras')  # Replace with the correct filename

def process_image(file):
    img = Image.open(file)
    img_seq = img.getdata()
    img_array = np.array(img_seq)
    img_array = np.reshape(img_array, (1, 64, 64, 3)) / 255.0
    return img_array

def predict_class(img_array):
    prediction = model.predict(img_array)
    class_label = "Vehicle" if np.any(prediction > 0.5) else "Non-Vehicle"
    return class_label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        img_array = process_image(file)
        class_label = predict_class(img_array)
        return render_template('index.html', result=class_label)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
