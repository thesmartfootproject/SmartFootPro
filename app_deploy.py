"""
Simplified Foot Deformity Classification API for Render Deployment
"""

import os
import io
import random
import time
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFootClassifier:
    """Simple foot deformity classifier for deployment"""
    
    def __init__(self):
        self.class_names = ['Normal', 'Flatfoot', 'Foot Ulcer', 'Hallux Valgus']
        logger.info(f"🔧 Initializing Simple Foot Classifier")
    
    def predict(self, image):
        """Simulate prediction with random results"""
        try:
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Random prediction for demo
            predicted_class = random.randint(0, len(self.class_names) - 1)
            confidence = random.uniform(0.7, 0.95)
            
            # Generate probabilities
            probabilities = [random.uniform(0.01, 0.3) for _ in self.class_names]
            probabilities[predicted_class] = confidence
            
            inference_time = time.time() - start_time
            
            result = {
                "prediction": self.class_names[predicted_class],
                "confidence": round(confidence * 100, 2),
                "class_id": predicted_class,
                "probabilities": {
                    class_name: round(prob * 100, 2) 
                    for class_name, prob in zip(self.class_names, probabilities)
                },
                "inference_time": round(inference_time, 3),
                "model_info": {
                    "name": "Demo Model",
                    "version": "1.0.0",
                    "device": "cpu"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize classifier
classifier = SimpleFootClassifier()

@app.route('/')
def home():
    return jsonify({
        "message": "Foot Deformity AI API is running!",
        "status": "healthy",
        "endpoints": ["/api/health", "/api/predict", "/api/model-info"]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """Main prediction endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Load image
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
        # Make prediction
        result = classifier.predict(image)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/recommendation', methods=['POST'])
def get_recommendation():
    """Get medical recommendation"""
    try:
        data = request.get_json()
        prediction = data.get('prediction')
        if not prediction:
            return jsonify({'error': 'No prediction provided'}), 400

        pred_class = prediction.get('prediction')
        confidence = prediction.get('confidence')
        
        recommendations = {
            "Normal": f"Based on {confidence}% confidence, continue regular foot care and monitoring.",
            "Flatfoot": f"With {confidence}% confidence, consider orthotic support and podiatrist consultation.",
            "Foot Ulcer": f"URGENT: {confidence}% confidence requires immediate medical attention.",
            "Hallux Valgus": f"Based on {confidence}% confidence, consider conservative treatment options."
        }
        
        recommendation = recommendations.get(pred_class, "Consult with a medical professional.")
        return jsonify({"recommendation": recommendation})
        
    except Exception as e:
        return jsonify({"error": "Unable to generate recommendation"}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "Demo Foot Classifier",
        "version": "1.0.0",
        "device": "cpu",
        "model_loaded": True,
        "classes": classifier.class_names,
        "description": "Demo model for deployment testing"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)