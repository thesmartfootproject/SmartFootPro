"""
Foot Deformity Classification API
Professional backend for multi-class foot deformity detection
"""

import os
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import logging
import time
from pathlib import Path
import base64
from datetime import datetime
import json
from threading import Lock
import requests
from dotenv import load_dotenv
import traceback
import os
import glob
import PyPDF2
import docx
import csv
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib

# MONAI imports
try:
    from monai.networks.nets import DenseNet121
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("⚠️ MONAI not available. Please install: pip install monai")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ANALYTICS_LOG = 'analytics_log.jsonl'
ANALYTICS_LOCK = Lock()

# Temporary hardcoded API key for testing
OPENROUTER_API_KEY = "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"

# Comment out the environment variable loading for now
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set. Please add it to your .env file.")

# --- RAG Knowledge Base Setup ---
KNOWLEDGE_BASE_DIR = r'C:\Users\DELL\Pictures\Amit data\Foot deformity Multi-Class\KnowledgeBase'
EMBED_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 400  # characters per chunk
TOP_K = 3

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL)

# Store passages and their embeddings
passages = []
embeddings = []
index = None # Initialize index to None

# Helper: chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def compute_kb_hash(kb_folder):
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(kb_folder):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            hash_md5.update(fname.encode())
            with open(fpath, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Helper: load and chunk all files
def load_knowledge_base():
    global passages, embeddings, index
    passages = []
    # PDFs
    for pdf_path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, '*.pdf')):
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text() or ''
                    for chunk in chunk_text(text):
                        passages.append(chunk)
        except Exception as e:
            print(f'Error reading PDF {pdf_path}:', e)
    # DOCX
    for docx_path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, '*.docx')):
        try:
            doc = docx.Document(docx_path)
            full_text = '\n'.join([p.text for p in doc.paragraphs])
            for chunk in chunk_text(full_text):
                passages.append(chunk)
        except Exception as e:
            print(f'Error reading DOCX {docx_path}:', e)
    # CSV
    for csv_path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, '*.csv')):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        for chunk in chunk_text(cell):
                            passages.append(chunk)
        except Exception as e:
            print(f'Error reading CSV {csv_path}:', e)
    # Compute knowledge base hash
    kb_hash = compute_kb_hash(KNOWLEDGE_BASE_DIR)
    hash_file = "kb_hash.txt"
    emb_file = "embeddings.npy"
    faiss_file = "faiss_index.faiss"

    # Check if all files exist and hash matches
    if (os.path.exists(hash_file) and os.path.exists(emb_file) and os.path.exists(faiss_file)):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()
        if saved_hash == kb_hash:
            print("Knowledge base unchanged. Loading embeddings and index from disk...")
            emb = np.load(emb_file)
            embeddings.clear()
            embeddings.extend(emb)
            index = faiss.read_index(faiss_file)
            return

    # Otherwise, recompute
    if passages:
        print(f'Embedding {len(passages)} passages...')
        emb = embedder.encode(passages, show_progress_bar=True, convert_to_numpy=True)
        embeddings.clear()
        embeddings.extend(emb)
        # Build FAISS index
        d = emb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings))
        print('Knowledge base loaded and indexed.')
        # Save to disk
        np.save(emb_file, emb)
        faiss.write_index(index, faiss_file)
        with open(hash_file, "w") as f:
            f.write(kb_hash)
    else:
        print('No passages found in knowledge base.')
        embeddings.clear()
        index = None

# Load KB on startup
load_knowledge_base()

class FootDeformityClassifier:
    """Professional foot deformity classification using MONAI DenseNet"""
    
    def __init__(self, model_path="models/monai_densenet_efficient.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Normal', 'Flatfoot', 'Foot Ulcer', 'Hallux Valgus']
        self.model = None
        self.transform = self._create_transform()
        
        logger.info(f"🔧 Initializing Foot Deformity Classifier")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Classes: {', '.join(self.class_names)}")
        
        # Load the model
        self.load_model(model_path)
    
    def _create_transform(self):
        """Create image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load the trained MONAI DenseNet model"""
        try:
            if not MONAI_AVAILABLE:
                raise ImportError("MONAI not available")
            
            # Create MONAI DenseNet121 model
            self.model = DenseNet121(
                spatial_dims=2,
                in_channels=3,
                out_channels=len(self.class_names),
                pretrained=False
            )
            
            # Load trained weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info(f"✅ Model loaded successfully from {model_path}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
                logger.info("🔄 Using randomly initialized model (for demo)")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Predict foot condition from image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return {"error": "Failed to preprocess image"}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            inference_time = time.time() - start_time
            
            # Prepare results
            result = {
                "prediction": self.class_names[predicted_class],
                "confidence": round(confidence * 100, 2),
                "class_id": predicted_class,
                "probabilities": {
                    class_name: round(prob * 100, 2) 
                    for class_name, prob in zip(self.class_names, probabilities[0].cpu().numpy())
                },
                "inference_time": round(inference_time, 3),
                "model_info": {
                    "name": "MONAI DenseNet121",
                    "version": "1.0.0",
                    "device": str(self.device)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🔍 Prediction: {result['prediction']} ({result['confidence']}%)")
            # Log analytics
            self.log_analytics(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def log_analytics(self, result):
        try:
            with ANALYTICS_LOCK:
                with open(ANALYTICS_LOG, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + '\n')
        except Exception as e:
            logger.error(f"Failed to log analytics: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize classifier
classifier = FootDeformityClassifier()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "device": str(classifier.device),
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

@app.route('/api/predict-base64', methods=['POST'])
def predict_base64_endpoint():
    """Prediction endpoint for base64 encoded images"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
        # Make prediction
        result = classifier.predict(image)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in predict-base64 endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": "MONAI DenseNet121",
        "version": "1.0.0",
        "device": str(classifier.device),
        "model_loaded": classifier.model is not None,
        "classes": classifier.class_names,
        "input_size": "224x224",
        "description": "Multi-class foot deformity classification model"
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({
        "classes": classifier.class_names,
        "count": len(classifier.class_names)
    })

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Return real analytics from logged predictions"""
    try:
        with ANALYTICS_LOCK:
            if not os.path.exists(ANALYTICS_LOG):
                return jsonify({
                    'totalAnalyses': 0,
                    'classifications': {},
                    'accuracy': None,
                    'averageResponseTime': None,
                    'history': []
                })
            with open(ANALYTICS_LOG, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        history = [json.loads(line) for line in lines]
        total = len(history)
        class_counts = {}
        total_time = 0
        for entry in history:
            pred = entry.get('prediction')
            class_counts[pred] = class_counts.get(pred, 0) + 1
            total_time += entry.get('inference_time', 0)
        avg_time = round(total_time / total, 3) if total else None
        # If you have ground truth, you can compute accuracy. Here, just return None.
        return jsonify({
            'totalAnalyses': total,
            'classifications': class_counts,
            'averageResponseTime': avg_time,
            'history': history[-100:]  # last 100 for frontend
        })
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/timeseries', methods=['GET'])
def analytics_timeseries():
    """Return time series of predictions (daily counts per class)"""
    try:
        with ANALYTICS_LOCK:
            if not os.path.exists(ANALYTICS_LOG):
                return jsonify({"series": {}})
            with open(ANALYTICS_LOG, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        history = [json.loads(line) for line in lines]
        # Group by date and class
        from collections import defaultdict
        series = defaultdict(lambda: defaultdict(int))
        for entry in history:
            ts = entry.get('timestamp')
            pred = entry.get('prediction')
            if ts and pred:
                date = ts[:10]  # YYYY-MM-DD
                series[date][pred] += 1
        # Convert to dict
        result = {date: dict(classes) for date, classes in series.items()}
        return jsonify({"series": result})
    except Exception as e:
        logger.error(f"Analytics timeseries error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/confidence', methods=['GET'])
def analytics_confidence():
    """Return confidence distribution per class"""
    try:
        with ANALYTICS_LOCK:
            if not os.path.exists(ANALYTICS_LOG):
                return jsonify({"confidence": {}})
            with open(ANALYTICS_LOG, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        history = [json.loads(line) for line in lines]
        from collections import defaultdict
        conf = defaultdict(list)
        for entry in history:
            pred = entry.get('prediction')
            confidence = entry.get('confidence')
            if pred and confidence is not None:
                conf[pred].append(confidence)
        # Optionally, bin confidences for histogram
        binned = {}
        for cls, vals in conf.items():
            bins = [0]*10
            for v in vals:
                idx = min(int(v//10), 9)
                bins[idx] += 1
            binned[cls] = bins
        return jsonify({"confidence": binned})
    except Exception as e:
        logger.error(f"Analytics confidence error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/errors', methods=['GET'])
def analytics_errors():
    """Return failed or low-confidence predictions (below 60%)"""
    try:
        with ANALYTICS_LOCK:
            if not os.path.exists(ANALYTICS_LOG):
                return jsonify({"errors": []})
            with open(ANALYTICS_LOG, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        history = [json.loads(line) for line in lines]
        errors = [entry for entry in history if entry.get('confidence', 100) < 60 or 'error' in entry]
        return jsonify({"errors": errors[-100:]})
    except Exception as e:
        logger.error(f"Analytics errors error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/download', methods=['GET'])
def analytics_download():
    """Download analytics as CSV"""
    import csv
    from flask import Response
    try:
        with ANALYTICS_LOCK:
            if not os.path.exists(ANALYTICS_LOG):
                return Response('No data', mimetype='text/csv')
            with open(ANALYTICS_LOG, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        history = [json.loads(line) for line in lines]
        if not history:
            return Response('No data', mimetype='text/csv')
        # Prepare CSV
        fieldnames = list(history[0].keys())
        def generate():
            writer = csv.DictWriter(io.StringIO(), fieldnames=fieldnames)
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)
            for row in history:
                writer.writerow(row)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)
        return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=analytics.csv"})
    except Exception as e:
        logger.error(f"Analytics download error: {e}")
        return Response(f'Error: {e}', mimetype='text/csv')

# --- AI Recommendation Endpoint (DeepSeek via OpenRouter) ---

@app.route('/api/recommendation', methods=['POST'])
def get_recommendation():
    try:
        data = request.get_json()
        prediction = data.get('prediction')
        if not prediction:
            return jsonify({'error': 'No prediction provided'}), 400

        pred_class = prediction.get('prediction')
        confidence = prediction.get('confidence')
        
        # Professional medical recommendations
        recommendations = {
            "Normal": f"Based on the {confidence}% confidence normal classification, continue regular foot hygiene and monitoring. Schedule annual podiatric check-ups for preventive care.",
            "Flatfoot": f"With {confidence}% confidence flatfoot detection, recommend orthotic evaluation, supportive footwear assessment, and physical therapy consultation for arch strengthening exercises.",
            "Foot Ulcer": f"URGENT: {confidence}% confidence ulcer detection requires immediate medical attention. Seek wound care specialist, maintain strict foot hygiene, and monitor for infection signs.",
            "Hallux Valgus": f"Based on {confidence}% confidence bunion detection, consider conservative management with proper footwear, toe spacers, and orthopedic consultation for treatment options."
        }
        
        recommendation = recommendations.get(pred_class, "Consult with a medical professional for comprehensive evaluation and personalized treatment plan.")
        return jsonify({"recommendation": recommendation})
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({"error": "Unable to generate recommendation"}), 500

@app.route('/api/voice-assistant', methods=['POST'])
def voice_assistant():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        print('No message provided to voice assistant endpoint')
        return jsonify({'error': 'No message provided'}), 400

    api_key = 'sk_bab4c8af6acc84443800d79b5bf5d879424e5494192f1f5d'
    agent_id = 'P39r1B8PJCGBBZL54HdP'

    # 1. Get AI text response from ElevenLabs Conversational AI
    try:
        chat_url = f'https://api.elevenlabs.io/v1/agents/{agent_id}/chat'
        chat_headers = {
            'xi-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        chat_payload = {
            'messages': [
                {'role': 'user', 'content': user_message}
            ]
        }
        chat_resp = requests.post(chat_url, headers=chat_headers, json=chat_payload, timeout=30)
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()
        ai_text = chat_data['messages'][-1]['content'] if 'messages' in chat_data and chat_data['messages'] else 'Sorry, no response.'
    except Exception as e:
        print('Voice Assistant ElevenLabs chat error:', e)
        print(traceback.format_exc())
        if 'chat_resp' in locals():
            print('Chat response text:', getattr(chat_resp, 'text', 'No response text'))
        return jsonify({'error': f'Failed to get AI response from ElevenLabs: {str(e)}'}), 500

    # 2. Get TTS audio from ElevenLabs
    try:
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        tts_headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        tts_payload = {
            "text": ai_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        tts_resp = requests.post(tts_url, headers=tts_headers, json=tts_payload, timeout=30)
        tts_resp.raise_for_status()
        audio_bytes = tts_resp.content
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print('Voice Assistant ElevenLabs TTS error:', e)
        print(traceback.format_exc())
        if 'tts_resp' in locals():
            print('TTS response text:', getattr(tts_resp, 'text', 'No response text'))
        return jsonify({"error": f"Failed to get audio from ElevenLabs: {str(e)}", "text": ai_text}), 500

    return jsonify({"answer": ai_text, "audio": audio_b64, "connected": True})

@app.route('/api/rag-assistant', methods=['POST'])
def rag_assistant():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    if not passages or not embeddings or index is None:
        return jsonify({'error': 'Knowledge base not loaded'}), 500
    # Embed question
    q_emb = embedder.encode([question], convert_to_numpy=True)
    # Retrieve top passages
    D, I = index.search(q_emb, TOP_K)
    retrieved = [passages[i] for i in I[0] if i < len(passages)]
    context = '\n'.join(retrieved)
    # Build improved prompt for DeepSeek
    prompt = (
        "You are a helpful orthopaedic assistant. "
        "Given the following context, answer the user's question in a clear, precise, and human-friendly way. "
        "Do not mention 'provided context', 'based on the context', or similar phrases. "
        "If listing steps or items, use concise bullet points.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    # Call DeepSeek via OpenRouter
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 400
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        choice = data.get("choices", [{}])[0].get("message", {})
        answer = choice.get("content") or choice.get("reasoning", "Sorry, I couldn't find an answer.")
        # Post-process: remove unwanted phrases
        for phrase in ["Based solely on the provided context", "Based on the provided context", "Based on the context", "provided context", "context provided", "Based on context"]:
            answer = answer.replace(phrase, "").strip()
        return jsonify({"answer": answer, "context": context})
    except Exception as e:
        print('RAG DeepSeek error:', e)
        return jsonify({'error': f'Failed to get answer from DeepSeek: {str(e)}'}), 500

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    api_key = 'sk_bab4c8af6acc84443800d79b5bf5d879424e5494192f1f5d'
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)
    try:
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        tts_headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        tts_payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        tts_resp = requests.post(tts_url, headers=tts_headers, json=tts_payload, timeout=30)
        tts_resp.raise_for_status()
        audio_bytes = tts_resp.content
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({"audio": audio_b64})
    except Exception as e:
        print('TTS ElevenLabs error:', e)
        if 'tts_resp' in locals():
            print('TTS response text:', getattr(tts_resp, 'text', 'No response text'))
        return jsonify({"error": f"Failed to get audio from ElevenLabs: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 







