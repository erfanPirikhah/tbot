import logging
from database.database import engine, Base
from database.models import User

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api_router import api_router
from app.core import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API")

app = FastAPI(
    title="Trading Bot API",
    description="API for the Intelligent Crypto Trading Bot",
    version="4.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Trading Bot API starting up...")
    logger.info("🔌 Initializing database connection...")

    import time
    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Test the database connection by attempting to create tables
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database connection established successfully!")
            logger.info("📋 Database tables created/verified")
            break  # Exit the loop if successful
        except Exception as e:
            retry_count += 1
            logger.warning(f"⚠️ Database connection failed on attempt {retry_count}, retrying in 2 seconds... Error: {e}")
            time.sleep(2)  # Wait 2 seconds before retrying
    else:
        # This else clause executes if the while loop completes without breaking
        logger.error(f"❌ Failed to connect to database after {max_retries} attempts")
        raise Exception(f"Could not connect to database after {max_retries} attempts")

@app.get("/")
async def root():
    return {"message": "Trading Bot API is running", "version": "4.0.0", "docs": "/docs"}
