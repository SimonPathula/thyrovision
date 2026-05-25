import os
import warnings

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from backend.routes import router, load_model
from utils.logger import logger

@asynccontextmanager
async def lifespan(_app):

    # Startup
    logger.info("Loading model during startup...")
    load_model()

    yield

    # Shutdown
    logger.info("Server shutting down...")

app = FastAPI(title="Thyroid Cancer Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Include Routes
app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting Thyroid Cancer Detection API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

