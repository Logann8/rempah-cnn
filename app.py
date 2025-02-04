from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Memuat model yang telah dilatih
MODEL_PATH = r'D:\Tugas\Pemodelan\UAS Pemodelan\rempah_cnn_project\models\rempah_cnn_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Kelas rempah-rempah
class_names = ['jahe', 'kencur', 'kunyit', 'lengkuas']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Simpan gambar sementara
        filepath = "temp.jpg"
        file.save(filepath)
        
        # Load dan preprocess gambar
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Hapus gambar sementara
        os.remove(filepath)
        
        return jsonify({'class': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
