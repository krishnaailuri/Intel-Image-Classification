from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("intel_image_classification_best_model.keras")

# Class names must match training
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

IMG_HEIGHT, IMG_WIDTH = 150, 150

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save uploaded file
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            # Preprocess and predict
            img = preprocess_image(filepath)
            preds = model.predict(img)
            pred_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)

            return render_template('result.html', pred=pred_class, confidence=confidence, image_file=file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
