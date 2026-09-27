"""Pydantic request and response schemas for the churn prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single-customer churn prediction request."""

    customer_id: Optional[str] = Field(None, description="Optional customer UUID")
    tenure_months: int = Field(..., ge=0, description="Months as a customer")
    contract_type: str = Field(..., description="Month-to-month | One year | Two year")
    total_spending_3m: float = Field(..., ge=0, description="Total spend last 3 months")
    avg_monthly_charge: float = Field(..., ge=0, description="Average monthly charge")
    payment_failure_rate: float = Field(..., ge=0, le=1, description="Fraction of failed payments")
    transaction_count_3m: int = Field(..., ge=0, description="Transaction count last 3 months")
    support_tickets_3m: int = Field(..., ge=0, description="Support tickets last 3 months")
    avg_resolution_days: float = Field(
        ..., ge=0, description="Average ticket resolution time (days)"
    )
    avg_sentiment: float = Field(
        ..., ge=-1.0, le=1.0, description="Average support sentiment score"
    )


class PredictResponse(BaseModel):
    """Single-customer churn prediction response."""

    customer_id: Optional[str] = None
    prediction: int = Field(..., description="1 = likely churn, 0 = no churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    timestamp: datetime


class BatchPredictRequest(BaseModel):
    """Batch churn prediction request."""

    customers: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    """Batch churn prediction response."""

    predictions: List[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    environment: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    stage: str
    tracking_uri: str
