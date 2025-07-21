import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Award, Clock, Target, Users, Zap, Download, Filter, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement, // Register for Line chart
  LineElement,  // Register for Line chart
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Analytics = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [timeseries, setTimeseries] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [errors, setErrors] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedClass, setSelectedClass] = useState('');
  const [dateRange, setDateRange] = useState({ from: '', to: '' });

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      try {
        const [main, ts, conf, errs] = await Promise.all([
          api.get('/api/analytics'),
          api.get('/api/analytics/timeseries'),
          api.get('/api/analytics/confidence'),
          api.get('/api/analytics/errors'),
        ]);
        setAnalyticsData(main.data);
        setTimeseries(ts.data.series);
        setConfidence(conf.data.confidence);
        setErrors(errs.data.errors);
        setError(null);
      } catch (err) {
        setError('Failed to fetch analytics');
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, []);

  const handleDownload = () => {
    window.open(api.defaults.baseURL + '/api/analytics/download', '_blank');
  };

  // Filtering helpers
  const filteredHistory = analyticsData?.history?.filter(item => {
    let match = true;
    if (selectedClass && item.prediction !== selectedClass) match = false;
    if (dateRange.from && new Date(item.timestamp) < new Date(dateRange.from)) match = false;
    if (dateRange.to && new Date(item.timestamp) > new Date(dateRange.to)) match = false;
    return match;
  }) || [];

  // Prepare time series chart data
  const timeLabels = timeseries ? Object.keys(timeseries).sort() : [];
  const classNames = analyticsData ? Object.keys(analyticsData.classifications) : [];
  const timeSeriesData = {
    labels: timeLabels,
    datasets: classNames.map((cls, i) => ({
      label: cls,
      data: timeLabels.map(date => timeseries?.[date]?.[cls] || 0),
      borderColor: ['#22c55e','#f59e0b','#ef4444','#8b5cf6'][i % 4],
      backgroundColor: ['#22c55e33','#f59e0b33','#ef444433','#8b5cf633'][i % 4],
      tension: 0.3,
      fill: false,
    })),
  };

  // Prepare confidence histogram data
  const confidenceData = {
    labels: Array.from({length: 10}, (_, i) => `${i*10}-${i*10+9}%`),
    datasets: classNames.map((cls, i) => ({
      label: cls,
      data: confidence?.[cls] || [],
      backgroundColor: ['#22c55e','#f59e0b','#ef4444','#8b5cf6'][i % 4],
    })),
  };

  if (loading) return <div className="text-center py-16">Loading analytics...</div>;
  if (error) return <div className="text-center text-red-600 py-16">{error}</div>;
  if (!analyticsData) return null;

  const { totalAnalyses, classifications, averageResponseTime } = analyticsData;

  const classificationData = {
    labels: Object.keys(classifications),
    datasets: [
      {
        data: Object.values(classifications),
        backgroundColor: [
          '#22c55e', // success
          '#f59e0b', // warning
          '#ef4444', // danger
          '#8b5cf6', // purple
        ],
        borderWidth: 0,
      },
    ],
  };

  const stats = [
    {
      title: 'Total Analyses',
      value: totalAnalyses?.toLocaleString() || 0,
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      title: 'Avg Response Time',
      value: averageResponseTime ? `${averageResponseTime}s` : 'N/A',
      icon: Clock,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    ...Object.entries(classifications).map(([cls, count], i) => ({
      title: `${cls} Count`,
      value: count,
      icon: [Target, Award, Zap, TrendingUp][i % 4],
      color: ['text-green-600','text-yellow-600','text-red-600','text-purple-600'][i % 4],
      bgColor: ['bg-green-100','bg-yellow-100','bg-red-100','bg-purple-100'][i % 4],
    })),
  ];

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
            Analytics Dashboard
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Real-time insights from actual analyses
          </p>
        </div>
        {/* Filters and Download */}
        <div className="flex flex-wrap gap-4 items-center justify-between mb-4">
          <div className="flex gap-2 items-center">
            <Filter className="w-5 h-5 text-gray-500" />
            <select className="border rounded px-2 py-1" value={selectedClass} onChange={e => setSelectedClass(e.target.value)}>
              <option value="">All Classes</option>
              {classNames.map(cls => <option key={cls} value={cls}>{cls}</option>)}
            </select>
            <input type="date" className="border rounded px-2 py-1" value={dateRange.from} onChange={e => setDateRange(r => ({...r, from: e.target.value}))} />
            <span>-</span>
            <input type="date" className="border rounded px-2 py-1" value={dateRange.to} onChange={e => setDateRange(r => ({...r, to: e.target.value}))} />
          </div>
          <button onClick={handleDownload} className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700">
            <Download className="w-4 h-4" /> Download CSV
          </button>
        </div>
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="card"
        >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                    <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  </div>
                  <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                    <Icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                    </div>
              </motion.div>
            );
          })}
                  </div>
                  
        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Time Series Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="card"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Predictions Over Time</h2>
              <BarChart3 className="w-6 h-6 text-gray-400" />
                    </div>
            <div className="h-80">
              <Line data={timeSeriesData} />
                    </div>
          </motion.div>
          {/* Confidence Histogram */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="card"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Confidence Distribution</h2>
              <Award className="w-6 h-6 text-gray-400" />
                  </div>
            <div className="h-80">
              <Bar data={confidenceData} options={{responsive:true, plugins:{legend:{display:true}}}} />
          </div>
        </motion.div>
        </div>
        {/* Recent Analyses Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Recent Analyses</h2>
            <TrendingUp className="w-6 h-6 text-gray-400" />
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-left">Time</th>
                  <th className="px-2 py-1 text-left">Prediction</th>
                  <th className="px-2 py-1 text-left">Confidence</th>
                  <th className="px-2 py-1 text-left">Inference Time</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory && filteredHistory.length > 0 ? filteredHistory.slice().reverse().map((item, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="px-2 py-1 whitespace-nowrap">{new Date(item.timestamp).toLocaleString()}</td>
                    <td className="px-2 py-1 font-medium">{item.prediction}</td>
                    <td className="px-2 py-1">{item.confidence}%</td>
                    <td className="px-2 py-1">{item.inference_time}s</td>
                  </tr>
                )) : (
                  <tr><td colSpan={4} className="text-center py-4 text-gray-500">No analyses yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
        {/* Low Confidence/Failed Predictions Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Low Confidence / Failed Predictions</h2>
            <AlertCircle className="w-6 h-6 text-gray-400" />
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-left">Time</th>
                  <th className="px-2 py-1 text-left">Prediction</th>
                  <th className="px-2 py-1 text-left">Confidence</th>
                  <th className="px-2 py-1 text-left">Inference Time</th>
                  <th className="px-2 py-1 text-left">Error</th>
                </tr>
              </thead>
              <tbody>
                {errors && errors.length > 0 ? errors.slice().reverse().map((item, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="px-2 py-1 whitespace-nowrap">{new Date(item.timestamp).toLocaleString()}</td>
                    <td className="px-2 py-1 font-medium">{item.prediction || '-'} </td>
                    <td className="px-2 py-1">{item.confidence !== undefined ? item.confidence + '%' : '-'}</td>
                    <td className="px-2 py-1">{item.inference_time !== undefined ? item.inference_time + 's' : '-'}</td>
                    <td className="px-2 py-1 text-red-600">{item.error || '-'}</td>
                  </tr>
                )) : (
                  <tr><td colSpan={5} className="text-center py-4 text-gray-500">No low-confidence or failed predictions.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Analytics;
