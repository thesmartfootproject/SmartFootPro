/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost', 'images.unsplash.com'],
    unoptimized: true,
  },
  env: {
    OPENROUTER_API_KEY: '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',
    OPENROUTER_MODEL: 'deepseek/deepseek-r1-0528:free',
  },
  async rewrites() {
    return [
      {
        source: '/api/predict',
        destination: 'http://localhost:8000/predict',
      },
      {
        source: '/api/health',
        destination: 'http://localhost:8000/health',
      },
      {
        source: '/api/recommendation',
        destination: 'http://localhost:5000/api/recommendation',
      },
    ];
  },
}

module.exports = nextConfig
