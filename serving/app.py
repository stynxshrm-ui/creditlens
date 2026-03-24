"""
serving/app.py

CreditLens FastAPI scoring endpoint.

Endpoints:
    POST /v1/score          — score a single loan application
    GET  /v1/health         — liveness check
    GET  /v1/model/info     — current champion model metadata
    GET  /v1/portfolio/risk-summary — portfolio PD distribution
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import duckdb
import logging
from pathlib import Path

from serving.schemas import (
    ScoreRequest, ScoreResponse, ModelInfo,
    HealthResponse, RiskTier, UpliftSegment
)
from serving.model_store import model_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/creditlens.duckdb")
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("Loading models...")
    model_store.load()
    logger.info("Models loaded — API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="CreditLens API",
    description="Credit risk scoring — PD prediction, uplift modelling, SHAP explainability",
    version=API_VERSION,
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {
        "name":    "CreditLens API",
        "version": API_VERSION,
        "docs":    "/docs",
        "health":  "/v1/health",
    }

@app.get("/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        models_loaded=model_store.is_loaded,
        version=API_VERSION,
    )


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """
    Score a single loan application.
    Returns PD score, credit score, risk tier, uplift score, and segment.
    """
    if not model_store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        features = request.model_dump()

        # Derive computed features if not provided
        if features.get("loan_to_income") is None and features["annual_inc"] > 0:
            features["loan_to_income"] = round(
                features["loan_amnt"] / features["annual_inc"], 4
            )
        if features.get("payment_to_income") is None and features["annual_inc"] > 0:
            features["payment_to_income"] = round(
                features["installment"] * 12 / features["annual_inc"], 4
            )

        # Derive risk flags if not provided
        if features.get("high_dti_flag") is None:
            features["high_dti_flag"] = int(features["dti"] > 28)
        if features.get("high_util_flag") is None and features.get("revol_util"):
            features["high_util_flag"] = int(features["revol_util"] > 75)
        if features.get("dual_stress_flag") is None:
            features["dual_stress_flag"] = int(
                features.get("high_dti_flag", 0) == 1
                and features.get("high_util_flag", 0) == 1
            )

        result = model_store.score(features)

        return ScoreResponse(
            pd_score=result["pd_score"],
            credit_score=result["credit_score"],
            risk_tier=RiskTier(result["risk_tier"]),
            uplift_score=result.get("uplift_score"),
            segment=UpliftSegment(result.get("segment", "unknown")),
            shap_values=None,   # Week 6 extension
            model_version=result["model_version"],
            champion_model=result["champion_model"],
        )

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/model/info", response_model=ModelInfo)
def model_info():
    """Return current champion model metadata."""
    if not model_store.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    info = model_store.model_info
    return ModelInfo(
        champion_model=info.get("champion_model", "XGBoost"),
        model_version=info.get("model_version", "unknown"),
        training_vintage=info.get("training_vintage", "2007-2016"),
        gini=info.get("gini", 0),
        ks=info.get("ks", 0),
        calibration_error=info.get("calibration_error", 0),
        registered_at=str(info.get("registered_at", "")),
    )


@app.get("/v1/portfolio/risk-summary")
def portfolio_risk_summary():
    """Return current portfolio PD distribution by risk tier."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        result = con.execute("""
            SELECT
                grade                               AS risk_grade,
                COUNT(*)                            AS loan_count,
                ROUND(AVG(o.default_flag)*100, 2)  AS actual_default_rate_pct,
                ROUND(AVG(b.dti), 2)               AS avg_dti,
                ROUND(AVG(l.loan_amnt), 0)         AS avg_loan_amount
            FROM loans l
            JOIN borrowers b USING (loan_id)
            JOIN outcomes  o USING (loan_id)
            GROUP BY grade
            ORDER BY grade
        """).fetchdf()
        con.close()
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))