import os

# Pydantic v2 moved BaseSettings to pydantic-settings; fall back gracefully
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]  # Pydantic v1 fallback


class Settings(BaseSettings):
    """
    Configuration variables for the Backend API.
    Used for overriding settings dynamically using Environment Variables.
    """
    PROJECT_NAME: str = "Deepfake Detection API"
    API_V1_STR: str = "/api/v1"
    
    # Model Configurations
    MODEL_TYPE: str = "efficientnet_b0" # Options: custom_cnn, efficientnet_b0
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "checkpoints/best_model.pth")
    CONFIDENCE_THRESHOLD: float = 0.50
    
    # Enable explainability (Grad-CAM takes extra compute time)
    ENABLE_GRADCAM: bool = True

    # Optional: Groq API key for Llama 4 Vision (free at console.groq.com)
    GROQ_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        
settings = Settings()
