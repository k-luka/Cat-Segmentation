from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load the model
# model = tf.keras.models.load_model('../unet_model_final.h5') # Commented out because model not yet trained

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process image
    result = process_image(filepath)
    
    return jsonify({
        'original': f'/static/uploads/{filename}',
        'processed': f'/static/uploads/processed_{filename}'
    })

def process_image(filepath):
    # Add your image processing code here
    # Use the model to generate segmentation
    return True

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)