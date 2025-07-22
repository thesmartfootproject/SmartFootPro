import os
import requests
from pathlib import Path

def download_model():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "monai_densenet_efficient.pth"
    
    if not model_path.exists():
        print("⚠️ Model file not found. Using dummy model for demo.")
        # Create a dummy file or download from a URL if you have one
        model_path.touch()
    
    return str(model_path)

if __name__ == "__main__":
    download_model()