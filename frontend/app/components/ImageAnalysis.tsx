'use client';

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  MagnifyingGlassIcon, 
  AdjustmentsHorizontalIcon,
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon
} from '@heroicons/react/24/outline';

interface ImageAnalysisProps {
  imageUrl: string;
  analysisResult?: any;
}

export default function ImageAnalysis({ imageUrl, analysisResult }: ImageAnalysisProps) {
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showAnnotations, setShowAnnotations] = useState(true);
  const imageRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };

  const resetView = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <MagnifyingGlassIcon className="w-5 h-5 mr-2 text-primary-600" />
          Image Analysis
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowAnnotations(!showAnnotations)}
            className={`btn-secondary text-sm ${showAnnotations ? 'bg-primary-100 text-primary-700' : ''}`}
          >
            <AdjustmentsHorizontalIcon className="w-4 h-4 mr-1" />
            Annotations
          </button>
        </div>
      </div>

      <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ height: '400px' }}>
        {/* Image Container */}
        <div
          ref={imageRef}
          className="relative w-full h-full cursor-move"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <img
            src={imageUrl}
            alt="Foot analysis"
            className="absolute inset-0 w-full h-full object-contain transition-transform duration-200"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
              transformOrigin: 'center'
            }}
            draggable={false}
          />

          {/* Analysis Annotations */}
          {showAnnotations && analysisResult && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 pointer-events-none"
            >
              {/* Confidence Indicator */}
              <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-3 py-2 rounded-lg text-sm">
                <div className="font-semibold">
                  {analysisResult.predicted_class.replace('_', ' ').toUpperCase()}
                </div>
                <div className="text-xs opacity-90">
                  {(analysisResult.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>

              {/* Analysis Regions (Mock) */}
              {analysisResult.predicted_class !== 'normal' && (
                <>
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-20 h-20 border-2 border-red-500 rounded-full animate-pulse opacity-75"></div>
                    <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-2 py-1 rounded text-xs">
                      Area of Interest
                    </div>
                  </div>
                </>
              )}

              {/* Measurement Lines (Mock) */}
              <svg className="absolute inset-0 w-full h-full">
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
                  </marker>
                </defs>
                <line x1="20%" y1="80%" x2="80%" y2="80%" 
                  stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />
                <text x="50%" y="85%" textAnchor="middle" fill="#3b82f6" fontSize="12">
                  Arch Height Analysis
                </text>
              </svg>
            </motion.div>
          )}
        </div>

        {/* Zoom Controls */}
        <div className="absolute bottom-4 right-4 flex flex-col space-y-2">
          <button
            onClick={handleZoomIn}
            className="bg-white shadow-lg rounded-lg p-2 hover:bg-gray-50 transition-colors"
            title="Zoom In"
          >
            <ArrowsPointingOutIcon className="w-4 h-4 text-gray-600" />
          </button>
          <button
            onClick={handleZoomOut}
            className="bg-white shadow-lg rounded-lg p-2 hover:bg-gray-50 transition-colors"
            title="Zoom Out"
          >
            <ArrowsPointingInIcon className="w-4 h-4 text-gray-600" />
          </button>
          <button
            onClick={resetView}
            className="bg-white shadow-lg rounded-lg p-2 hover:bg-gray-50 transition-colors text-xs"
            title="Reset View"
          >
            1:1
          </button>
        </div>

        {/* Zoom Level Indicator */}
        <div className="absolute top-4 right-4 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-xs">
          {Math.round(zoom * 100)}%
        </div>
      </div>

      {/* Image Info */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Format:</span>
          <span className="ml-2 font-medium">JPEG</span>
        </div>
        <div>
          <span className="text-gray-500">Resolution:</span>
          <span className="ml-2 font-medium">224x224</span>
        </div>
        <div>
          <span className="text-gray-500">Quality:</span>
          <span className="ml-2 font-medium text-green-600">High</span>
        </div>
        <div>
          <span className="text-gray-500">Processing:</span>
          <span className="ml-2 font-medium">Complete</span>
        </div>
      </div>

      {/* Analysis Tips */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <h4 className="text-sm font-semibold text-blue-900 mb-2">Analysis Features:</h4>
        <ul className="text-xs text-blue-800 space-y-1">
          <li>• Click and drag to pan around the image</li>
          <li>• Use zoom controls to examine details</li>
          <li>• Toggle annotations to see analysis regions</li>
          <li>• Red circles indicate areas of medical interest</li>
        </ul>
      </div>
    </div>
  );
}
