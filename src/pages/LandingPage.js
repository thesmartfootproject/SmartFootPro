import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Shield, 
  Eye, 
  CheckCircle, 
  AlertTriangle,
  ArrowRight,
  Star,
  Award,
  Clock
} from 'lucide-react';

const LandingPage = () => {
  const features = [
    {
      icon: Brain,
      title: 'MONAI DenseNet121',
      description: 'State-of-the-art medical AI architecture specifically designed for medical image analysis with 98.8% accuracy.',
      color: 'from-primary-500 to-primary-600'
    },
    {
      icon: Eye,
      title: 'Multi-Classification',
      description: 'Simultaneously detects Normal, Flatfoot, Foot Ulcer, and Hallux Valgus conditions from medical images.',
      color: 'from-secondary-500 to-secondary-600'
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Lightning-fast inference in ~96ms per image with confidence scores and probability distributions.',
      color: 'from-warning-500 to-warning-600'
    },
    {
      icon: Shield,
      title: 'Medical Grade',
      description: '99.6% Hallux Valgus recall rate ensuring virtually no missed cases in critical diagnoses.',
      color: 'from-success-500 to-success-600'
    }
  ];

  const models = [
    {
      name: 'MONAI DenseNet121',
      type: 'Medical AI Framework',
      accuracy: 98.8,
      hvRecall: 99.6,
      precision: 94.3,
      speed: 96,
      recommended: true,
      description: 'Specialized medical AI model using MONAI framework with DenseNet121 architecture, optimized for medical image analysis.'
    },
    {
      name: 'Vision Transformer',
      type: 'Attention-Based Model',
      accuracy: 96.2,
      hvRecall: 97.8,
      precision: 92.1,
      speed: 142,
      recommended: false,
      description: 'State-of-the-art transformer architecture adapted for medical image classification with excellent interpretability.'
    },
    {
      name: 'ResNet50',
      type: 'Convolutional Neural Network',
      accuracy: 94.5,
      hvRecall: 95.2,
      precision: 89.7,
      speed: 78,
      recommended: false,
      description: 'Proven CNN architecture with residual connections, providing robust baseline performance.'
    }
  ];

  const classifications = [
    {
      icon: CheckCircle,
      name: 'Normal',
      description: 'Healthy foot structure with proper arch formation',
      accuracy: 98.2,
      color: 'text-success-600 bg-success-100'
    },
    {
      icon: Eye,
      name: 'Flatfoot',
      description: 'Collapsed or low arch condition (pes planus)',
      accuracy: 97.1,
      color: 'text-warning-600 bg-warning-100'
    },
    {
      icon: Shield,
      name: 'Foot Ulcer',
      description: 'Open wounds or diabetic ulcers requiring medical attention',
      accuracy: 95.8,
      color: 'text-error-600 bg-error-100'
    },
    {
      icon: AlertTriangle,
      name: 'Hallux Valgus',
      description: 'Bunion deformity of the big toe joint',
      accuracy: 99.6,
      color: 'text-secondary-600 bg-secondary-100'
    }
  ];

  const stats = [
    { label: 'Overall Accuracy', value: '98.8%', icon: Award },
    { label: 'Hallux Valgus Recall', value: '99.6%', icon: Star },
    { label: 'Inference Time', value: '96ms', icon: Clock },
    { label: 'Classifications', value: '4', icon: Brain }
  ];

  return (
    <div className="pt-4">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-4 lg:py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 leading-tight">
                Professional{' '}
                <span className="gradient-text">Medical Image</span>{' '}
                Analysis
              </h1>
              <p className="text-xl text-gray-600 mt-6 leading-relaxed">
                Advanced AI-powered foot deformity detection using state-of-the-art deep learning models for clinical-grade image analysis
              </p>
              
              {/* Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
                {stats.map((stat, index) => {
                  const Icon = stat.icon;
                  return (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      className="card text-center"
                    >
                      <Icon className="h-8 w-8 text-primary-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                      <div className="text-sm text-gray-600">{stat.label}</div>
                    </motion.div>
                  );
                })}
              </div>

              <div className="flex flex-col sm:flex-row gap-4 mt-8">
                <Link to="/dashboard" className="btn-primary inline-flex items-center justify-center">
                  <Brain className="h-5 w-5 mr-2" />
                  Start Analysis
                  <ArrowRight className="h-5 w-5 ml-2" />
                </Link>
                <Link to="/documentation" className="btn-secondary inline-flex items-center justify-center">
                  View Documentation
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative"
            >
              <div className="card p-8 animate-float">
                <div className="text-center">
                  <div className="w-24 h-24 bg-gradient-to-r from-primary-600 to-secondary-600 rounded-2xl mx-auto mb-6 flex items-center justify-center animate-pulse-glow">
                    <Brain className="h-12 w-12 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-4">AI-Powered Analysis</h3>
                  <p className="text-gray-600">
                    Upload your foot images and get instant, accurate analysis with detailed confidence scores and probability distributions.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Advanced Medical AI Features
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive comparison of state-of-the-art deep learning architectures for medical image classification
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card-hover text-center"
                >
                  <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-2xl mx-auto mb-6 flex items-center justify-center`}>
                    <Icon className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Models Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Advanced AI Models
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Choose from multiple state-of-the-art deep learning architectures optimized for medical image analysis
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {models.map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className={`card-hover relative ${model.recommended ? 'ring-2 ring-primary-500' : ''}`}
              >
                {model.recommended && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
                      Recommended
                    </span>
                  </div>
                )}

                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-primary-600 to-secondary-600 rounded-xl flex items-center justify-center mr-4">
                    <Brain className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{model.name}</h3>
                    <p className="text-sm text-gray-600">{model.type}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center p-3 bg-gray-50 rounded-xl">
                    <div className="text-2xl font-bold text-gray-900">{model.accuracy}%</div>
                    <div className="text-sm text-gray-600">Accuracy</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-xl">
                    <div className="text-2xl font-bold text-gray-900">{model.hvRecall}%</div>
                    <div className="text-sm text-gray-600">HV Recall</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-xl">
                    <div className="text-2xl font-bold text-gray-900">{model.precision}%</div>
                    <div className="text-sm text-gray-600">Precision</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-xl">
                    <div className="text-2xl font-bold text-gray-900">{model.speed}ms</div>
                    <div className="text-sm text-gray-600">Speed</div>
                  </div>
                </div>

                <p className="text-gray-600 mb-6 leading-relaxed">{model.description}</p>

                <div className="flex gap-3">
                  <Link to="/documentation" className="flex-1 btn-secondary text-center">
                    View Details
                  </Link>
                  <Link to="/dashboard" className="flex-1 btn-primary text-center">
                    Test Model
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Classifications Section */}
      <section className="py-20 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Supported Classifications
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive foot deformity detection covering four major conditions with clinical-grade accuracy
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {classifications.map((classification, index) => {
              const Icon = classification.icon;
              return (
                <motion.div
                  key={classification.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card-hover text-center"
                >
                  <div className={`w-20 h-20 ${classification.color} rounded-2xl mx-auto mb-6 flex items-center justify-center`}>
                    <Icon className="h-10 w-10" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">{classification.name}</h3>
                  <p className="text-gray-600 mb-4 leading-relaxed">{classification.description}</p>
                  <span className="inline-block bg-success-100 text-success-800 px-3 py-1 rounded-full text-sm font-semibold">
                    {classification.accuracy}% Accuracy
                  </span>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="card p-12"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
              Ready to Start Analyzing?
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Experience the power of AI-driven medical image analysis. Upload your foot images and get instant, accurate results with detailed insights.
            </p>
            <Link to="/dashboard" className="btn-primary inline-flex items-center text-lg px-8 py-4">
              <Brain className="h-6 w-6 mr-3" />
              Launch Dashboard
              <ArrowRight className="h-6 w-6 ml-3" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;