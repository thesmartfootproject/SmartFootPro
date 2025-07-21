import React from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  Download, 
  RotateCcw, 
  CheckCircle, 
  Footprints, 
  Bandage, 
  AlertTriangle,
  Clock,
  Brain
} from 'lucide-react';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const ResultsDisplay = ({ results, onReset, onDownload }) => {
  if (!results) return null;

  const getClassificationIcon = (classification) => {
    switch (classification) {
      case 'Normal':
        return CheckCircle;
      case 'Flatfoot':
        return Footprints;
      case 'Foot Ulcer':
        return Bandage;
      case 'Hallux Valgus':
        return AlertTriangle;
      default:
        return CheckCircle;
    }
  };

  const getClassificationColor = (classification) => {
    switch (classification) {
      case 'Normal':
        return 'from-success-500 to-success-600';
      case 'Flatfoot':
        return 'from-warning-500 to-warning-600';
      case 'Foot Ulcer':
        return 'from-error-500 to-error-600';
      case 'Hallux Valgus':
        return 'from-secondary-500 to-secondary-600';
      default:
        return 'from-primary-500 to-primary-600';
    }
  };

  const Icon = getClassificationIcon(results.prediction);
  const colorClass = getClassificationColor(results.prediction);

  // Chart data
  const chartData = {
    labels: Object.keys(results.probabilities),
    datasets: [
      {
        data: Object.values(results.probabilities),
        backgroundColor: [
          '#22c55e', // Normal - Green
          '#f59e0b', // Flatfoot - Orange
          '#ef4444', // Foot Ulcer - Red
          '#a855f7', // Hallux Valgus - Purple
        ],
        borderColor: '#ffffff',
        borderWidth: 3,
        hoverBorderWidth: 4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          usePointStyle: true,
          font: {
            size: 12,
            family: 'Inter',
            weight: '500'
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          }
        },
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8
      }
    },
    cutout: '60%'
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-6"
    >
      {/* Main Result */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2" />
          Analysis Results
        </h2>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`bg-gradient-to-r ${colorClass} text-white p-6 rounded-2xl mb-6 relative overflow-hidden`}
        >
          {/* Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent transform -skew-x-12" />
          </div>
          
          <div className="relative z-10 text-center">
            <Icon className="h-12 w-12 mx-auto mb-4" />
            <h3 className="text-3xl font-bold mb-2">{results.prediction}</h3>
            <p className="text-xl opacity-90 mb-2">
              Confidence: {results.confidence.toFixed(1)}%
            </p>
            <div className="flex items-center justify-center space-x-4 text-sm opacity-80">
              <div className="flex items-center space-x-1">
                <Brain className="h-4 w-4" />
                <span>{results.model.name}</span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4" />
                <span>~{results.model.inferenceTime}ms</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Probability Chart */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
            Probability Distribution
          </h3>
          <div className="h-80">
            <Doughnut data={chartData} options={chartOptions} />
          </div>
        </div>

        {/* Detailed Probabilities */}
        <div className="space-y-3">
          <h4 className="text-md font-semibold text-gray-900">Detailed Probabilities</h4>
          {Object.entries(results.probabilities).map(([classification, probability]) => {
            const ClassIcon = getClassificationIcon(classification);
            const isHighest = classification === results.prediction;
            
            return (
              <div
                key={classification}
                className={`flex items-center justify-between p-3 rounded-xl ${
                  isHighest ? 'bg-primary-50 border border-primary-200' : 'bg-gray-50'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <ClassIcon className={`h-5 w-5 ${
                    classification === 'Normal' ? 'text-success-600' :
                    classification === 'Flatfoot' ? 'text-warning-600' :
                    classification === 'Foot Ulcer' ? 'text-error-600' :
                    'text-secondary-600'
                  }`} />
                  <span className="font-medium text-gray-900">{classification}</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        classification === 'Normal' ? 'bg-success-500' :
                        classification === 'Flatfoot' ? 'bg-warning-500' :
                        classification === 'Foot Ulcer' ? 'bg-error-500' :
                        'bg-secondary-500'
                      }`}
                      style={{ width: `${probability}%` }}
                    />
                  </div>
                  <span className="font-semibold text-gray-900 w-12 text-right">
                    {probability.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 mt-6">
          <button
            onClick={onReset}
            className="btn-secondary flex-1 flex items-center justify-center"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            New Analysis
          </button>
          <button
            onClick={onDownload}
            className="btn-primary flex-1 flex items-center justify-center"
          >
            <Download className="h-4 w-4 mr-2" />
            Download Report
          </button>
        </div>
      </div>

      {/* Model Performance Info */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-xl">
            <div className="text-2xl font-bold text-gray-900">{results.model.accuracy}%</div>
            <div className="text-sm text-gray-600">Overall Accuracy</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-xl">
            <div className="text-2xl font-bold text-gray-900">{results.model.hvRecall}%</div>
            <div className="text-sm text-gray-600">HV Recall</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-xl">
            <div className="text-2xl font-bold text-gray-900">{results.model.precision}%</div>
            <div className="text-sm text-gray-600">Precision</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-xl">
            <div className="text-2xl font-bold text-gray-900">{results.model.inferenceTime}ms</div>
            <div className="text-sm text-gray-600">Inference Time</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ResultsDisplay;
