'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  ClockIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  DocumentArrowDownIcon,
  PrinterIcon,
  ShareIcon
} from '@heroicons/react/24/outline';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import toast from 'react-hot-toast';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

interface AnalysisResult {
  predicted_class: string;
  confidence: number;
  description: string;
  all_probabilities: Record<string, number>;
  inference_time: number;
  recommendations?: string[];
  recommendation?: {
    title: string;
    priority: string;
    immediate_actions: string[];
    follow_up: string[];
    lifestyle_tips: string[];
  };
}

interface ResultsPanelProps {
  result: AnalysisResult;
}

const conditionColors: Record<string, string> = {
  'normal': '#22c55e',
  'flatfoot': '#3b82f6',
  'foot_ulcer': '#ef4444',
  'hallux_valgus': '#8b5cf6'
};

const conditionInfo: Record<string, { severity: string; urgency: string; icon: any }> = {
  'normal': { severity: 'None', urgency: 'Routine', icon: CheckCircleIcon },
  'flatfoot': { severity: 'Mild to Moderate', urgency: 'Non-urgent', icon: InformationCircleIcon },
  'foot_ulcer': { severity: 'Moderate to Severe', urgency: 'Urgent', icon: ExclamationTriangleIcon },
  'hallux_valgus': { severity: 'Mild to Moderate', urgency: 'Non-urgent', icon: InformationCircleIcon }
};

export default function ResultsPanel({ result }: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'detailed' | 'recommendations'>('overview');

  // Prepare chart data
  const chartData = {
    labels: Object.keys(result.all_probabilities).map(key => 
      key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
    ),
    datasets: [
      {
        data: Object.values(result.all_probabilities).map(val => (val * 100).toFixed(1)),
        backgroundColor: Object.keys(result.all_probabilities).map(key => conditionColors[key] || '#6b7280'),
        borderWidth: 2,
        borderColor: '#ffffff'
      }
    ]
  };

  const barChartData = {
    labels: Object.keys(result.all_probabilities).map(key => 
      key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
    ),
    datasets: [
      {
        label: 'Confidence (%)',
        data: Object.values(result.all_probabilities).map(val => (val * 100).toFixed(1)),
        backgroundColor: Object.keys(result.all_probabilities).map(key => conditionColors[key] || '#6b7280'),
        borderRadius: 4,
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: {
          padding: 20,
          usePointStyle: true,
        }
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.parsed}%`;
          }
        }
      }
    }
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `Confidence: ${context.parsed.y}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          }
        }
      }
    }
  };

  const getConditionStatus = () => {
    const info = conditionInfo[result.predicted_class];
    const IconComponent = info?.icon || InformationCircleIcon;
    
    let statusColor = 'text-blue-600 bg-blue-100';
    if (result.predicted_class === 'foot_ulcer') statusColor = 'text-red-600 bg-red-100';
    else if (result.predicted_class === 'normal') statusColor = 'text-green-600 bg-green-100';
    
    return { info, IconComponent, statusColor };
  };

  const downloadReport = () => {
    const reportData = {
      timestamp: new Date().toISOString(),
      analysis: result,
      patient_id: 'DEMO-' + Date.now()
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `foot-analysis-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success('Report downloaded successfully');
  };

  const printReport = () => {
    window.print();
    toast.success('Print dialog opened');
  };

  const shareReport = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Foot Analysis Report',
          text: `Analysis Result: ${result.predicted_class} (${(result.confidence * 100).toFixed(1)}% confidence)`,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
      toast.success('Report link copied to clipboard');
    }
  };

  const { info, IconComponent, statusColor } = getConditionStatus();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <ChartBarIcon className="w-7 h-7 mr-3 text-primary-600" />
          Analysis Results
        </h2>
        <div className="flex space-x-2">
          <button onClick={downloadReport} className="btn-secondary text-sm flex items-center">
            <DocumentArrowDownIcon className="w-4 h-4 mr-1" />
            Download
          </button>
          <button onClick={printReport} className="btn-secondary text-sm flex items-center">
            <PrinterIcon className="w-4 h-4 mr-1" />
            Print
          </button>
          <button onClick={shareReport} className="btn-secondary text-sm flex items-center">
            <ShareIcon className="w-4 h-4 mr-1" />
            Share
          </button>
        </div>
      </div>

      {/* Main Result */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <IconComponent className={`w-8 h-8 ${statusColor.split(' ')[0]}`} />
            <div>
              <h3 className="text-xl font-semibold text-gray-900 capitalize">
                {result.predicted_class.replace('_', ' ')}
              </h3>
              <p className="text-gray-600">{result.description}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-primary-600">
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Severity:</span>
            <span className="ml-2 font-medium">{info?.severity || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-gray-500">Urgency:</span>
            <span className="ml-2 font-medium">{info?.urgency || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-gray-500">Processing:</span>
            <span className="ml-2 font-medium">{result.inference_time.toFixed(0)}ms</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'detailed', label: 'Detailed Analysis' },
            { id: 'recommendations', label: 'Recommendations' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === 'overview' && (
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-semibold mb-4">Probability Distribution</h4>
              <div className="h-64">
                <Doughnut data={chartData} options={chartOptions} />
              </div>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Confidence Levels</h4>
              <div className="h-64">
                <Bar data={barChartData} options={barChartOptions} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'detailed' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-4">Detailed Probability Breakdown</h4>
              <div className="space-y-3">
                {Object.entries(result.all_probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([condition, probability]) => (
                    <div key={condition} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div 
                          className="w-4 h-4 rounded-full"
                          style={{ backgroundColor: conditionColors[condition] || '#6b7280' }}
                        ></div>
                        <span className="font-medium capitalize">
                          {condition.replace('_', ' ')}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full"
                            style={{ 
                              width: `${probability * 100}%`,
                              backgroundColor: conditionColors[condition] || '#6b7280'
                            }}
                          ></div>
                        </div>
                        <span className="font-semibold text-gray-900 w-12 text-right">
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Technical Details</h4>
              <div className="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Model:</span>
                  <span className="font-medium">ResNet50 (Improved)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Processing Time:</span>
                  <span className="font-medium">{result.inference_time.toFixed(2)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Image Resolution:</span>
                  <span className="font-medium">224x224 pixels</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Analysis Date:</span>
                  <span className="font-medium">{new Date().toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'recommendations' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-4">Clinical Recommendations</h4>
              {result.recommendation ? (
                <div className="space-y-4">
                  <div>
                    <div className="text-xl font-bold mb-1">{result.recommendation.title}</div>
                    <div className="text-sm text-gray-600 mb-2">Priority: <span className="font-semibold">{result.recommendation.priority}</span></div>
                  </div>
                  <div>
                    <div className="font-semibold mb-1">Immediate Actions:</div>
                    <ul className="list-disc ml-6">
                      {result.recommendation.immediate_actions.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="font-semibold mb-1">Follow Up:</div>
                    <ul className="list-disc ml-6">
                      {result.recommendation.follow_up.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="font-semibold mb-1">Lifestyle Tips:</div>
                    <ul className="list-disc ml-6">
                      {result.recommendation.lifestyle_tips.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                <p className="text-gray-600 italic">No specific recommendations available for this analysis.</p>
              )}
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <ExclamationTriangleIcon className="w-6 h-6 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h5 className="font-semibold text-yellow-800 mb-2">Important Medical Disclaimer</h5>
                  <p className="text-yellow-700 text-sm">
                    This AI analysis is for informational purposes only and should not replace professional medical diagnosis or treatment. 
                    Always consult with qualified healthcare professionals, such as podiatrists or orthopedic specialists, for proper medical evaluation and treatment planning.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}
