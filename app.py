from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import cv2

app = Flask(__name__)

# Load your custom trained model
model = load_model('trained_model/object_detector.h5')

# Get class names from your training data structure
CLASS_NAMES = sorted(os.listdir('raw_data'))  # Adjust path if needed
CONFIDENCE_THRESHOLD = 0.6  # Adjust based on your needs

def preprocess_image(image):
    # Resize and normalize as done during training
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_object():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify(success=False, error="No image provided")
            
        image_file = request.files['image']
        image = Image.open(image_file.stream)
        
        # Convert to OpenCV format (BGR) and process
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)[0]
        max_confidence = np.max(predictions)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        
        # Prepare response
        result = {
            'object': predicted_class if max_confidence >= CONFIDENCE_THRESHOLD else 'no_object_detected',
            'confidence': float(max_confidence),
            'all_predictions': {name: float(conf) for name, conf in zip(CLASS_NAMES, predictions)}
        }
        
        return jsonify(success=True, result=result)
        
    except Exception as e:
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)