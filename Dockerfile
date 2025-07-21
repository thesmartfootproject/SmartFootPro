# Production Dockerfile for Hallux Valgus Detection API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY production_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r production_requirements.txt

# Copy application files
COPY production_backend.py .
COPY production_config.py .
COPY start_production.py .

# Create models directory
RUN mkdir -p models

# Copy model file if it exists (optional)
# COPY models/monai_densenet_efficient.pth models/

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "start_production.py"]
