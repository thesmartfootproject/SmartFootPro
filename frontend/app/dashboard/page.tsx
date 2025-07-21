'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CloudArrowUpIcon, 
  PhotoIcon, 
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import ImageAnalysis from '../components/ImageAnalysis';
import AIAssistant from '../components/AIAssistant';
import ResultsPanel from '../components/ResultsPanel';

interface Recommendation {
  title: string;
  priority: string;
  immediate_actions: string[];
  follow_up: string[];
  lifestyle_tips: string[];
}

interface AnalysisResult {
  predicted_class: string;
  confidence: number;
  description: string;
  all_probabilities: Record<string, number>;
  inference_time: number;
  recommendation?: Recommendation;
}

export default function Dashboard() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showAIAssistant, setShowAIAssistant] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast.error('File size must be less than 10MB');
        return;
      }

      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file');
        return;
      }

      setUploadedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      toast.success('Image uploaded successfully');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    multiple: false
  });

  const analyzeImage = async () => {
    if (!uploadedImage) {
      toast.error('Please upload an image first');
      return;
    }

    setIsAnalyzing(true);
    try {
      // Simulate API call to our ML model
      const formData = new FormData();
      formData.append('image', uploadedImage);

      // Replace this with your real API call
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      const prediction = await response.json();
      console.log('Prediction response:', prediction);

      // Defensive check for prediction validity
      if (!response.ok || !prediction || prediction.error || !prediction.prediction) {
        toast.error(prediction.error || 'Prediction failed. Please try again.');
        setIsAnalyzing(false);
        return;
      }

      // Map backend keys to frontend keys for compatibility
      const mappedPrediction = {
        ...prediction,
        predicted_class: prediction.prediction || prediction.predicted_class || prediction.class || '',
        all_probabilities: prediction.probabilities || prediction.all_probabilities || {},
        confidence: typeof prediction.confidence === 'number' ? prediction.confidence / 100 : prediction.confidence, // backend returns 0-100, frontend expects 0-1
        inference_time: prediction.inference_time || 0,
        description: prediction.description || '',
      };

      // Fetch recommendation from backend
      let recommendation = undefined;
      try {
        const recResp = await fetch('/api/recommendation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prediction }),
        });
        const recData = await recResp.json();
        recommendation = recData.recommendation;
      } catch (err) {
        recommendation = undefined;
      }

      setAnalysisResult({
        ...mappedPrediction,
        recommendation,
      });
      toast.success('Analysis completed successfully');
    } catch (error) {
      toast.error('Analysis failed. Please try again.');
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setUploadedImage(null);
    setImagePreview(null);
    setAnalysisResult(null);
    setShowAIAssistant(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Foot Deformity Analysis Dashboard
          </h1>
          <p className="text-gray-600">
            Upload a foot image for AI-powered deformity classification and analysis
          </p>
        </div>

        {/* Developer Test Button */}
        {process.env.NODE_ENV !== 'production' && (
          <div className="mb-4">
            <button
              className="btn-secondary px-4 py-2 rounded text-sm"
              onClick={async () => {
                try {
                  const resp = await fetch('/api/recommendation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prediction: { prediction: 'Flatfoot' } })
                  });
                  const data = await resp.json();
                  if (data.recommendation) {
                    toast.success(`Test Recommendation: ${data.recommendation.title}`);
                  } else {
                    toast.error('No recommendation returned');
                  }
                } catch (err) {
                  toast.error('Test failed: ' + err);
                }
              }}
            >
              Test AI Recommendation
            </button>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="card mb-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <PhotoIcon className="w-6 h-6 mr-2 text-primary-600" />
                Image Upload
              </h2>
              
              {!imagePreview ? (
                <div
                  {...getRootProps()}
                  className={`upload-area ${isDragActive ? 'dragover' : ''}`}
                >
                  <input {...getInputProps()} />
                  <CloudArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-700 mb-2">
                    {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                  </p>
                  <p className="text-gray-500 mb-4">or click to select a file</p>
                  <div className="text-sm text-gray-400">
                    Supports: JPEG, PNG, BMP, TIFF (max 10MB)
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <img
                      src={imagePreview}
                      alt="Uploaded foot image"
                      className="w-full h-64 object-contain bg-gray-100 rounded-lg"
                    />
                    <button
                      onClick={resetAnalysis}
                      className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                    >
                      ×
                    </button>
                  </div>
                  
                  <div className="flex space-x-4">
                    <button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="btn-primary flex-1 flex items-center justify-center"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="spinner mr-2"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <CpuChipIcon className="w-5 h-5 mr-2" />
                          Analyze Image
                        </>
                      )}
                    </button>
                    
                    {analysisResult && (
                      <button
                        onClick={() => setShowAIAssistant(true)}
                        className="btn-secondary flex items-center"
                      >
                        <DocumentTextIcon className="w-5 h-5 mr-2" />
                        AI Assistant
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Analysis Results */}
            <AnimatePresence>
              {analysisResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <ResultsPanel result={analysisResult} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Analysis Statistics
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Model Accuracy</span>
                  <span className="font-semibold text-green-600">96.9%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Avg. Processing Time</span>
                  <span className="font-semibold text-blue-600">&lt;50ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Conditions Detected</span>
                  <span className="font-semibold text-purple-600">4 Types</span>
                </div>
              </div>
            </div>

            {/* Condition Guide */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Detectable Conditions
              </h3>
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-green-500 rounded-full mt-1.5"></div>
                  <div>
                    <div className="font-medium text-gray-900">Normal</div>
                    <div className="text-sm text-gray-600">Healthy foot structure</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mt-1.5"></div>
                  <div>
                    <div className="font-medium text-gray-900">Flatfoot</div>
                    <div className="text-sm text-gray-600">Collapsed arch condition</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-red-500 rounded-full mt-1.5"></div>
                  <div>
                    <div className="font-medium text-gray-900">Foot Ulcer</div>
                    <div className="text-sm text-gray-600">Skin lesions and wounds</div>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-purple-500 rounded-full mt-1.5"></div>
                  <div>
                    <div className="font-medium text-gray-900">Hallux Valgus</div>
                    <div className="text-sm text-gray-600">Bunion deformity</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Image Guidelines
              </h3>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-start">
                  <CheckCircleIcon className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  Clear, well-lit images
                </li>
                <li className="flex items-start">
                  <CheckCircleIcon className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  Full foot visible in frame
                </li>
                <li className="flex items-start">
                  <CheckCircleIcon className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  Minimal background clutter
                </li>
                <li className="flex items-start">
                  <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                  Avoid blurry or dark images
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* AI Assistant Modal */}
        <AnimatePresence>
          {showAIAssistant && analysisResult && (
            <AIAssistant
              result={analysisResult}
              onClose={() => setShowAIAssistant(false)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
