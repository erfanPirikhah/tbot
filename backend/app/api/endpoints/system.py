from fastapi import APIRouter
import platform
import sys
from datetime import datetime

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Check if the API is running and return system info
    """
    return {
        "status": "online",
        "timestamp": datetime.now(),
        "system": platform.system(),
        "python_version": sys.version
    }
