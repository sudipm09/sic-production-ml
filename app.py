from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sic-api")

MODEL_PATH = Path(__file__).resolve().parent / "sic_model.joblib"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("SIC model loaded successfully")
    except Exception:
        model = None
        logger.exception("SIC model loading failed")

    yield
    model = None


app = FastAPI(
    title="SIC Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class SICInput(BaseModel):
    albedo: float = Field(ge=0, le=1)
    SST: float
    U: float
    V: float
    SWD: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    return {"status": "ready"}


@app.post("/predict")
def predict(data: SICInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    try:
        features = np.array([[
            data.albedo,
            data.SST,
            data.U,
            data.V,
            data.SWD
        ]])

        prediction = model.predict(features)

        logger.info("Prediction completed")

        return {
            "predicted_sic": float(prediction[0])
        }

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

