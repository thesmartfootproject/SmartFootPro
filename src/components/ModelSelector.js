import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Award, Clock, CheckCircle } from 'lucide-react';

const ModelSelector = ({ selectedModel, onModelChange, models }) => {
  const getModelIcon = (modelId) => {
    switch (modelId) {
      case 'monai':
        return Brain;
      case 'vit':
        return Zap;
      case 'resnet':
        return Award;
      default:
        return Brain;
    }
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
        <Brain className="h-5 w-5 mr-2" />
        Select AI Model
      </h2>
      
      <div className="space-y-3">
        {Object.entries(models).map(([modelId, model]) => {
          const Icon = getModelIcon(modelId);
          const isSelected = selectedModel === modelId;
          
          return (
            <motion.div
              key={modelId}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onModelChange(modelId)}
              className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                isSelected
                  ? 'border-primary-500 bg-primary-50 shadow-md'
                  : 'border-gray-200 hover:border-primary-300 hover:bg-primary-25'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-lg ${
                    isSelected 
                      ? 'bg-primary-500 text-white' 
                      : 'bg-gray-100 text-gray-600'
                  }`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <h3 className="font-semibold text-gray-900">{model.name}</h3>
                      {model.recommended && (
                        <span className="bg-success-100 text-success-800 px-2 py-1 rounded-full text-xs font-medium">
                          Recommended
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                    
                    {/* Performance Metrics */}
                    <div className="grid grid-cols-4 gap-3 mt-3">
                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-900">{model.accuracy}%</div>
                        <div className="text-xs text-gray-600">Accuracy</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-900">{model.hvRecall}%</div>
                        <div className="text-xs text-gray-600">HV Recall</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-900">{model.precision}%</div>
                        <div className="text-xs text-gray-600">Precision</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-900">{model.inferenceTime}ms</div>
                        <div className="text-xs text-gray-600">Speed</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {isSelected && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="text-primary-500"
                  >
                    <CheckCircle className="h-6 w-6" />
                  </motion.div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
      
      {/* Model Comparison Note */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          <strong>Note:</strong> You can switch between models at any time. 
          The image will be automatically re-analyzed with the selected model.
        </p>
      </div>
    </div>
  );
};

export default ModelSelector;
