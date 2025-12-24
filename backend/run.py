import uvicorn
import os
import sys
import logging

# Add the current directory to python path so we can import app and lib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Startup")

if __name__ == "__main__":
    logger.info("🚀 Starting Trading Bot API with database initialization...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
