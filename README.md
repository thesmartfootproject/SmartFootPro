# Foot Deformity AI - Professional Classification System
A professional-grade web application for AI-powered foot deformity classification using advanced deep learning models.

<img width="1917" height="905" alt="Screenshot 2025-08-09 145858" src="https://github.com/user-attachments/assets/3c96c00e-8446-4d3a-a5d2-d8a44fd731b4" />

## 🚀 Features

- **Multi-class Classification**: Detect Normal, Flatfoot, Foot Ulcer, and Hallux Valgus conditions
- **High Accuracy**: 98.8% overall accuracy with 99.6% Hallux Valgus recall
- **Fast Inference**: ~96ms average response time
- **Modern UI**: Professional React frontend with Tailwind CSS
- **RESTful API**: Clean Flask backend with comprehensive endpoints
- **Real-time Analysis**: Instant image processing and results
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🏗️ Architecture

### Backend (Python/Flask)
- **Framework**: Flask with CORS support
- **AI Model**: MONAI DenseNet121 , VIT and Resnet50 for medical image analysis
- **Image Processing**: PIL and torchvision transforms
- **API**: RESTful endpoints with JSON responses

### Frontend (React)
- **Framework**: React 18 with React Router
- **Styling**: Tailwind CSS with custom design system
- **Animations**: Framer Motion for smooth interactions
- **Charts**: Chart.js for analytics visualization
- **File Upload**: React Dropzone for drag-and-drop functionality

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup
```bash
# Install Node.js dependencies
npm install

# Start the development server
npm start
```

The frontend will run on `http://localhost:3000`

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
FLASK_ENV=development
FLASK_DEBUG=True
MODEL_PATH=models/monai_densenet_efficient.pth
```

### Model Setup
Ensure the trained model file is placed in the `models/` directory:
```
models/
└── monai_densenet_efficient.pth
```

## 📚 API Documentation

### Endpoints

#### POST `/api/predict`
Analyze foot image for deformity classification

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Foot image (JPG, PNG, BMP, TIFF)

**Response:**
```json
{
  "prediction": "Normal",
  "confidence": 95.2,
  "class_id": 0,
  "probabilities": {
    "Normal": 95.2,
    "Flatfoot": 2.1,
    "Foot Ulcer": 1.8,
    "Hallux Valgus": 0.9
  },
  "inference_time": 0.096,
  "model_info": {
    "name": "MONAI DenseNet121",
    "version": "1.0.0",
    "device": "cpu"
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### POST `/api/predict-base64`
Analyze base64 encoded image

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

#### GET `/api/health`
Check API health and system status

#### GET `/api/model-info`
Get model information and capabilities

#### GET `/api/classes`
Get available classification classes

## 🎯 Usage

### Web Interface
1. Open the application in your browser
2. Upload a foot image using drag-and-drop or file picker
3. Click "Analyze Image" to process
4. View results with confidence scores and breakdown

### API Integration
```python
import requests

# Upload image file
with open('foot_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 98.8% |
| Hallux Valgus Recall | 99.6% |
| Precision | 97.2% |
| Average Inference Time | ~96ms |
| Model Architecture | MONAI DenseNet121 |

## 🏥 Supported Conditions

1. **Normal**: Healthy foot structure
2. **Flatfoot**: Pes planus condition
3. **Foot Ulcer**: Diabetic foot ulcers
4. **Hallux Valgus**: Bunions

## 🛠️ Development

### Project Structure
```
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── package.json          # Node.js dependencies
├── tailwind.config.js    # Tailwind configuration
├── public/               # Static assets
├── src/                  # React source code
│   ├── components/       # Reusable components
│   ├── pages/           # Page components
│   ├── App.js           # Main app component
│   └── index.js         # Entry point
└── models/              # AI model files
```

### Available Scripts
```bash
# Frontend
npm start          # Start development server
npm build          # Build for production
npm test           # Run tests

# Backend
python app.py      # Start Flask server
```

## 🔒 Security & Privacy

- Images are processed locally and not stored
- No personal data is collected or transmitted
- API endpoints validate file types and sizes
- CORS configured for secure cross-origin requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation page in the application
- Review the API health endpoint for system status

## 🙏 Acknowledgments

- MONAI framework for medical AI capabilities
- React and Tailwind CSS communities
- Medical imaging research community

---

**Note**: This system is designed for educational and research purposes. For clinical use, please ensure proper validation and regulatory compliance. 
