"""
Simplified Foot Deformity Classification API
For testing without ML dependencies
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleFootClassifier:
    """Simple foot deformity classifier for testing"""
    
    def __init__(self):
        self.class_names = ['Normal', 'Flatfoot', 'Foot Ulcer', 'Hallux Valgus']
        logger.info(f"🔧 Initializing Simple Foot Classifier")
        logger.info(f"🎯 Classes: {', '.join(self.class_names)}")
    
    def predict(self, image):
        """Mock prediction for testing"""
        try:
            start_time = time.time()
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Generate mock prediction
            predicted_class = random.choice(self.class_names)
            confidence = random.uniform(75.0, 99.0)
            
            inference_time = time.time() - start_time
            
            # Generate mock probabilities
            probabilities = {}
            remaining_prob = 100 - confidence
            for class_name in self.class_names:
                if class_name == predicted_class:
                    probabilities[class_name] = round(confidence, 2)
                else:
                    prob = random.uniform(0.1, remaining_prob / (len(self.class_names) - 1))
                    probabilities[class_name] = round(prob, 2)
                    remaining_prob -= prob
            
            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: round(v * 100 / total, 2) for k, v in probabilities.items()}
            
            result = {
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
                "class_id": self.class_names.index(predicted_class),
                "probabilities": probabilities,
                "inference_time": round(inference_time, 3),
                "model_info": {
                    "name": "Simple Test Model",
                    "version": "1.0.0",
                    "device": "cpu"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🔍 Mock Prediction: {result['prediction']} ({result['confidence']}%)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize classifier
classifier = SimpleFootClassifier()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "endpoints": [
            "/api/predict",
            "/api/health",
            "/api/model-info",
            "/api/classes"
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """Main prediction endpoint"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Supported: PNG, JPG, JPEG, BMP, TIFF"}), 400
        
        # Load and process image
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
        logger.error(f"❌ Error in predict endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "Simple Test Model",
        "version": "1.0.0",
        "device": "cpu",
        "model_loaded": True,
        "classes": classifier.class_names,
        "input_size": "224x224",
        "description": "Simple test model for demonstration"
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({
        "classes": classifier.class_names,
        "count": len(classifier.class_names)
    })

@app.route('/api/recommendation', methods=['POST'])
def get_recommendation():
    """Mock recommendation endpoint"""
    try:
        data = request.get_json()
        prediction = data.get('prediction')
        if not prediction:
            return jsonify({'error': 'No prediction provided'}), 400

        pred_class = prediction.get('prediction', 'Unknown')
        confidence = prediction.get('confidence', 0)

        # Professional medical recommendations with structured format
        recommendations = {
            'Normal': {
                'title': '✅ Normal Foot Structure Detected',
                'priority': 'Routine Care',
                'immediate_actions': [
                    'Continue current foot care routine',
                    'Maintain healthy weight and activity level',
                    'Wear properly fitted, supportive footwear'
                ],
                'follow_up': [
                    'Schedule annual foot health check-ups',
                    'Monitor for any changes in foot structure or comfort',
                    'Maintain good foot hygiene practices'
                ],
                'lifestyle_tips': [
                    'Regular stretching and foot exercises',
                    'Choose appropriate footwear for activities',
                    'Keep feet clean and dry'
                ]
            },
            'Flatfoot': {
                'title': '⚠️ Flatfoot (Pes Planus) Detected',
                'priority': 'Moderate - Requires Attention',
                'immediate_actions': [
                    'Consult with a podiatrist or orthopedic specialist',
                    'Consider custom orthotic inserts or arch supports',
                    'Switch to supportive, motion-control footwear'
                ],
                'follow_up': [
                    'Physical therapy evaluation for foot strengthening',
                    'Regular monitoring of symptoms and progression',
                    'Follow-up appointment in 4-6 weeks'
                ],
                'lifestyle_tips': [
                    'Avoid prolonged standing on hard surfaces',
                    'Perform calf stretches and foot strengthening exercises',
                    'Maintain healthy weight to reduce foot stress'
                ]
            },
            'Foot Ulcer': {
                'title': '🚨 Foot Ulcer Detected - Urgent Care Required',
                'priority': 'HIGH PRIORITY - Immediate Medical Attention',
                'immediate_actions': [
                    'Seek immediate medical attention from wound care specialist',
                    'Keep wound clean and covered with sterile dressing',
                    'Avoid weight-bearing on affected foot if possible',
                    'Do NOT attempt self-treatment'
                ],
                'follow_up': [
                    'Daily wound assessment and dressing changes',
                    'Regular follow-up with healthcare provider',
                    'Monitor for signs of infection (redness, warmth, discharge)'
                ],
                'lifestyle_tips': [
                    'Strict blood sugar control if diabetic',
                    'Proper nutrition to support wound healing',
                    'Avoid smoking and alcohol which impair healing'
                ]
            },
            'Hallux Valgus': {
                'title': '👣 Hallux Valgus (Bunion) Detected',
                'priority': 'Moderate - Progressive Condition',
                'immediate_actions': [
                    'Consult with podiatrist for severity assessment',
                    'Switch to wide-toe box, low-heel footwear',
                    'Use bunion pads or toe spacers for comfort'
                ],
                'follow_up': [
                    'Regular monitoring of deformity progression',
                    'Physical therapy for toe mobility and strength',
                    'Surgical consultation if conservative treatment fails'
                ],
                'lifestyle_tips': [
                    'Avoid high heels and narrow, pointed shoes',
                    'Perform toe stretching and strengthening exercises',
                    'Apply ice for pain and inflammation relief'
                ]
            }
        }

        recommendation = recommendations.get(pred_class, 'Consult with a medical professional for proper diagnosis and treatment.')
        
        return jsonify({"recommendation": recommendation})
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": f"Failed to get recommendation: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Simple Foot Deformity AI Server...")
    print("📱 Frontend should be available at: http://localhost:3000")
    print("🔧 Backend API available at: http://localhost:5000")
    print("📚 API Documentation: http://localhost:3000/documentation")
    app.run(debug=True, host='0.0.0.0', port=5000) 