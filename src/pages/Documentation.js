import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, 
  Code, 
  BookOpen, 
  Zap,
  CheckCircle,
  AlertCircle,
  Info,
  Copy,
  ExternalLink
} from 'lucide-react';
import toast from 'react-hot-toast';

const Documentation = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  const tabs = [
    { id: 'overview', name: 'Overview', icon: BookOpen },
    { id: 'api', name: 'API Reference', icon: Code },
    { id: 'examples', name: 'Examples', icon: FileText },
    { id: 'models', name: 'Models', icon: Zap },
  ];

  const apiEndpoints = [
    {
      method: 'POST',
      path: '/api/predict',
      description: 'Analyze foot image for deformity classification',
      parameters: [
        { name: 'image', type: 'file', required: true, description: 'Foot image file (JPG, PNG, BMP, TIFF)' }
      ]
    },
    {
      method: 'POST',
      path: '/api/predict-base64',
      description: 'Analyze base64 encoded image',
      parameters: [
        { name: 'image', type: 'string', required: true, description: 'Base64 encoded image data' }
      ]
    },
    {
      method: 'GET',
      path: '/api/health',
      description: 'Check API health and system status',
      parameters: []
    },
    {
      method: 'GET',
      path: '/api/model-info',
      description: 'Get model information and capabilities',
      parameters: []
    },
    {
      method: 'GET',
      path: '/api/classes',
      description: 'Get available classification classes',
      parameters: []
    }
  ];

  const codeExamples = [
    {
      title: 'Python - File Upload',
      language: 'python',
      code: `import requests

url = "http://localhost:5000/api/predict"
files = {"image": open("foot_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")`
    },
    {
      title: 'JavaScript - File Upload',
      language: 'javascript',
      code: `const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.prediction);
  console.log('Confidence:', data.confidence);
});`
    },
    {
      title: 'cURL - File Upload',
      language: 'bash',
      code: `curl -X POST http://localhost:5000/api/predict \\
  -F "image=@foot_image.jpg" \\
  -H "Content-Type: multipart/form-data"`
    }
  ];

  const modelInfo = {
    name: 'MONAI DenseNet121',
    version: '1.0.0',
    architecture: 'DenseNet121 with MONAI framework',
    inputSize: '224x224 pixels',
    classes: ['Normal', 'Flatfoot', 'Foot Ulcer', 'Hallux Valgus'],
    performance: {
      accuracy: '98.8%',
      halluxValgusRecall: '99.6%',
      precision: '97.2%',
      inferenceTime: '~96ms'
    }
  };

  return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="space-y-8"
        >
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Documentation
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Complete guide to using the Foot Deformity AI API and understanding the system
          </p>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="min-h-[600px]">
          {activeTab === 'overview' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">System Overview</h2>
                <p className="text-gray-600 mb-6">
                  The Foot Deformity AI system is a professional-grade medical image analysis platform 
                  that uses advanced deep learning to classify foot conditions. Built with MONAI framework 
                  and optimized for clinical accuracy.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Key Features</h3>
                    <ul className="space-y-2">
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Multi-class foot deformity classification</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>High accuracy (98.8% overall)</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Fast inference (~96ms)</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>RESTful API interface</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span>Multiple image format support</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Supported Conditions</h3>
                    <ul className="space-y-2">
                      <li className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <span><strong>Normal:</strong> Healthy foot structure</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                        <span><strong>Flatfoot:</strong> Pes planus condition</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                        <span><strong>Foot Ulcer:</strong> Diabetic foot ulcers</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                        <span><strong>Hallux Valgus:</strong> Bunions</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Getting Started</h2>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-bold">
                      1
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">Start the API Server</h3>
                      <p className="text-gray-600">Run the Flask backend server on port 5000</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-bold">
                      2
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">Upload an Image</h3>
                      <p className="text-gray-600">Use the web interface or API to upload a foot image</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-bold">
                      3
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">Get Results</h3>
                      <p className="text-gray-600">Receive classification results with confidence scores</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'api' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">API Endpoints</h2>
                
                <div className="space-y-6">
                  {apiEndpoints.map((endpoint, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-6">
                      <div className="flex items-center space-x-3 mb-4">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          endpoint.method === 'GET' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {endpoint.method}
                        </span>
                        <code className="text-lg font-mono text-gray-900">{endpoint.path}</code>
                      </div>
                      
                      <p className="text-gray-600 mb-4">{endpoint.description}</p>
                      
                      {endpoint.parameters.length > 0 && (
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2">Parameters:</h4>
                          <div className="space-y-2">
                            {endpoint.parameters.map((param, paramIndex) => (
                              <div key={paramIndex} className="flex items-center space-x-4 text-sm">
                                <code className="font-mono bg-gray-100 px-2 py-1 rounded">
                                  {param.name}
                                </code>
                                <span className="text-gray-500">({param.type})</span>
                                <span className={param.required ? 'text-red-600' : 'text-gray-500'}>
                                  {param.required ? 'Required' : 'Optional'}
                                </span>
                                <span className="text-gray-600">{param.description}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              </motion.div>
          )}

          {activeTab === 'examples' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
        >
          <div className="card">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Code Examples</h2>
            
                <div className="space-y-6">
                  {codeExamples.map((example, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-between p-4 border-b border-gray-200">
                        <h3 className="font-semibold text-gray-900">{example.title}</h3>
                        <button
                          onClick={() => copyToClipboard(example.code)}
                          className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900"
                        >
                          <Copy className="w-4 h-4" />
                          <span>Copy</span>
                        </button>
                      </div>
                      <pre className="p-4 bg-gray-50 overflow-x-auto">
                        <code className="text-sm text-gray-800">{example.code}</code>
                      </pre>
                    </div>
                  ))}
            </div>
          </div>
        </motion.div>
          )}

          {activeTab === 'models' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Model Information</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Details</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Name:</span>
                        <span className="font-medium">{modelInfo.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Version:</span>
                        <span className="font-medium">{modelInfo.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Architecture:</span>
                        <span className="font-medium">{modelInfo.architecture}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Input Size:</span>
                        <span className="font-medium">{modelInfo.inputSize}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Overall Accuracy:</span>
                        <span className="font-medium text-green-600">{modelInfo.performance.accuracy}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Hallux Valgus Recall:</span>
                        <span className="font-medium text-blue-600">{modelInfo.performance.halluxValgusRecall}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Precision:</span>
                        <span className="font-medium text-purple-600">{modelInfo.performance.precision}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Inference Time:</span>
                        <span className="font-medium text-orange-600">{modelInfo.performance.inferenceTime}</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Supported Classes</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {modelInfo.classes.map((className, index) => (
                      <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                        <span className="font-medium text-gray-900">{className}</span>
                      </div>
                    ))}
                  </div>
                </div>
          </div>
        </motion.div>
          )}
      </div>
      </motion.div>
    </div>
  );
};

export default Documentation;
