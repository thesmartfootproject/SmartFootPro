import React from 'react';
import { motion } from 'framer-motion';
import { Image as ImageIcon } from 'lucide-react';

const ImagePreview = ({ image, className = '' }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative ${className}`}
    >
      <div className="relative bg-gray-100 rounded-lg overflow-hidden">
        <img
          src={image}
          alt="Uploaded image"
          className="w-full h-48 object-cover"
          onError={(e) => {
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        <div className="hidden absolute inset-0 bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500 text-sm">Failed to load image</p>
          </div>
        </div>
      </div>
      
      {/* Image overlay with info */}
      <div className="absolute top-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-medium">
        Preview
      </div>
    </motion.div>
  );
};

export default ImagePreview;
