"""
Production Configuration for Hallux Valgus Detection API
"""

import os
from pathlib import Path

class ProductionConfig:
    """Production configuration settings"""
    
    # Server Configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/monai_densenet_efficient.pth')
    DEVICE = os.environ.get('DEVICE', 'auto')  # 'auto', 'cpu', 'cuda'
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance Configuration
    THREADED = True
    PROCESSES = 1
    
    # Security Configuration (for production deployment)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check model file exists
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        # Validate port range
        if not (1 <= cls.PORT <= 65535):
            errors.append(f"Invalid port number: {cls.PORT}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {cls.LOG_LEVEL}")
        
        return errors

# Environment-specific configurations
class DevelopmentConfig(ProductionConfig):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(ProductionConfig):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

class ProductionDeploymentConfig(ProductionConfig):
    """Production deployment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    CORS_ORIGINS = ['https://yourdomain.com']  # Restrict CORS in production

# Configuration factory
def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'production').lower()
    
    if env == 'development':
        return DevelopmentConfig
    elif env == 'testing':
        return TestingConfig
    elif env == 'production':
        return ProductionDeploymentConfig
    else:
        return ProductionConfig
