import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import {
  Upload,
  Image as ImageIcon,
  Brain, 
  Clock, 
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Info
} from 'lucide-react';
import ImagePreview from '../components/ImagePreview';
import LoadingAnimation from '../components/LoadingAnimation';
import VoiceAssistant from '../components/VoiceAssistant';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

const Dashboard = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendation, setRecommendation] = useState(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
    const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage({
          file,
          preview: reader.result
        });
        setPrediction(null);
    };
    reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    multiple: false
  });

  const handlePredict = async () => {
    if (!selectedImage) {
      toast.error('Please select an image first');
      return;
    }
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedImage.file);
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setPrediction(data);
      toast.success('Analysis completed successfully!');
    } catch (error) {
      toast.error('Failed to analyze image');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPrediction(null);
    setRecommendation(null);
    setRecError(null);
  };

  // Fetch AI recommendation after prediction
  useEffect(() => {
    const fetchRecommendation = async () => {
      if (!prediction) return;
      setRecLoading(true);
      setRecError(null);
      setRecommendation(null);
      try {
        const response = await fetch('/api/recommendation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prediction }),
    });
        const data = await response.json();
        setRecommendation(data.recommendation || 'No recommendation available.');
      } catch (err) {
        setRecError('Failed to fetch recommendation.');
      } finally {
        setRecLoading(false);
      }
    };
    fetchRecommendation();
  }, [prediction]);

  // Helper for confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };
  const getConfidenceBgColor = (confidence) => {
    if (confidence >= 80) return 'bg-green-100';
    if (confidence >= 60) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 pb-24">
      {/* Hero Banner */}
      <div className="w-full bg-gradient-to-r from-blue-600 to-purple-600 py-10 px-4 mb-8 rounded-b-3xl shadow-lg">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-3 drop-shadow-lg">Foot Deformity AI Dashboard</h1>
          <p className="text-lg md:text-2xl text-blue-100 font-medium mb-2">Upload a foot image to analyze and classify potential deformities using advanced AI technology.</p>
        </div>
      </div>
      {/* Stepper */}
      <div className="max-w-3xl mx-auto flex items-center justify-center gap-4 mb-10 px-2">
        <div className="flex items-center gap-2">
          <span className="w-9 h-9 flex items-center justify-center rounded-full bg-primary-600 text-white font-bold shadow">1</span>
          <span className="font-medium text-gray-700">Upload Image</span>
        </div>
        <span className="w-8 h-0.5 bg-gray-300 mx-2" />
        <div className="flex items-center gap-2">
          <span className="w-9 h-9 flex items-center justify-center rounded-full bg-primary-600 text-white font-bold shadow">2</span>
          <span className="font-medium text-gray-700">Analyze</span>
        </div>
        <span className="w-8 h-0.5 bg-gray-300 mx-2" />
        <div className="flex items-center gap-2">
          <span className="w-9 h-9 flex items-center justify-center rounded-full bg-primary-600 text-white font-bold shadow">3</span>
          <span className="font-medium text-gray-700">View Results</span>
        </div>
      </div>
      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-10 px-4">
        {/* Left: Upload & Help */}
        <div className="space-y-8">
          {/* Upload Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-blue-100"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Upload Image</h2>
              <ImageIcon className="w-6 h-6 text-gray-400" />
            </div>
      <div
        {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 cursor-pointer ${
          isDragActive
                  ? 'border-primary-400 bg-primary-50'
                  : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">
                {isDragActive
                  ? 'Drop the image here...'
                  : 'Drag & drop an image here, or click to select'}
            </p>
              <p className="text-sm text-gray-500">
                Supports: JPG, PNG, BMP, TIFF (Max 10MB)
              </p>
            </div>
            {selectedImage && (
          <motion.div
                initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
                className="mt-6"
          >
                <ImagePreview image={selectedImage.preview} />
                <div className="flex space-x-3 mt-4">
                  <button
                    onClick={handlePredict}
                    disabled={isLoading}
                    className="btn-primary flex-1 flex items-center justify-center space-x-2"
                  >
                    {isLoading ? (
                      <LoadingAnimation />
                    ) : (
                      <>
                        <Brain className="w-4 h-4" />
                        <span>Analyze Image</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    className="btn-secondary flex items-center justify-center space-x-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                </div>
              </motion.div>
                )}
          </motion.div>
          {/* Help Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl shadow p-6 border border-blue-100 flex flex-col items-center text-center"
          >
            <Info className="w-10 h-10 text-primary-600 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-4">How to Use</h3>
            <ol className="list-decimal list-inside text-gray-700 text-base space-y-2 mb-4 text-left max-w-xs mx-auto">
              <li><span className="font-medium">Upload a clear foot image</span> (JPG, PNG, BMP, TIFF).</li>
              <li>Click <span className="font-semibold">Analyze Image</span> to run the AI model.</li>
              <li>View the <span className="font-semibold">predicted class</span> and <span className="font-semibold">confidence breakdown</span>.</li>
              <li>Use <span className="font-semibold">Reset</span> to try another image.</li>
            </ol>
            <div className="text-xs text-gray-500 mt-2">For best results, use high-quality, well-lit images.</div>
          </motion.div>
        </div>
        {/* Right: Results & Recommendation */}
        <div className="space-y-8">
          {/* Results Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-blue-100"
                >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Analysis Results</h2>
              <Brain className="w-6 h-6 text-gray-400" />
                  </div>
            {!prediction && !isLoading && (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <ImageIcon className="w-8 h-8 text-gray-400" />
                    </div>
                <p className="text-gray-500">Upload an image to see analysis results</p>
                    </div>
            )}
            {isLoading && (
              <div className="text-center py-12">
                <LoadingAnimation />
                <p className="text-gray-600 mt-4">Analyzing image...</p>
                  </div>
              )}
            {prediction && (
                <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Main Result */}
                <div className="text-center">
                  <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${getConfidenceBgColor(prediction.confidence)} ${getConfidenceColor(prediction.confidence)}`}>
                    {prediction.confidence >= 80 ? (
                      <CheckCircle className="w-4 h-4 mr-2" />
                    ) : (
                      <AlertCircle className="w-4 h-4 mr-2" />
                    )}
                    {prediction.prediction}
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mt-2">
                    {prediction.confidence}% Confidence
                  </p>
                  </div>
                {/* Confidence Breakdown */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Confidence Breakdown</h3>
                  <div className="space-y-3">
                    {Object.entries(prediction && prediction.probabilities ? prediction.probabilities : {}).map(([className, confidence]) => (
                      <div key={className} className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">{className}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-24 bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                className === prediction.prediction
                                  ? 'bg-primary-600'
                                  : 'bg-gray-300'
                              }`}
                              style={{ width: `${confidence}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium text-gray-900 w-12 text-right">
                            {confidence}%
                          </span>
                        </div>
                      </div>
                    ))}
                              </div>
                            </div>
                {/* Model Info */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Model:</span>
                    <span className="font-medium">{prediction.model_info.name}</span>
                              </div>
                  <div className="flex items-center justify-between text-sm mt-1">
                    <span className="text-gray-600">Inference Time:</span>
                    <span className="font-medium flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {prediction.inference_time}s
                              </span>
                  </div>
                      </div>
              </motion.div>
            )}
          </motion.div>
          {/* AI Recommendation Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl shadow p-6 border border-blue-100 min-h-[180px] flex flex-col"
          >
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-6 h-6 text-blue-500" />
              <div className="font-semibold text-blue-700 text-lg">AI Recommendation</div>
                      </div>
            <div className="flex-1 overflow-y-auto px-2 py-1">
              {recLoading ? (
                <div className="flex items-center justify-center h-full animate-pulse text-blue-500">
                  <LoadingAnimation />
                  <span className="ml-2">Generating recommendation...</span>
                      </div>
              ) : recError ? (
                <div className="text-red-500">{recError}</div>
              ) : (
                <div className="prose prose-blue max-w-none text-blue-900 text-base leading-relaxed">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                      a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" />
                    }}
                  >
                    {typeof recommendation === 'string' && recommendation.trim() ? recommendation : 'No recommendation available.'}
                  </ReactMarkdown>
                  </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
      {/* Floating Voice Assistant */}
      <VoiceAssistant />
    </div>
  );
};

export default Dashboard;
