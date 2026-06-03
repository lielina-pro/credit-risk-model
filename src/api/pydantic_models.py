"""
Pydantic models for the Credit Risk API.
Defines request and response schemas with validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Input features for a single customer transaction.
    All fields match the feature engineering pipeline output.
    """
    # Core transaction features
    Value: float = Field(..., gt=0, description="Transaction value in UGX")
    PricingStrategy: int = Field(
        ..., ge=0, le=4, description="Pricing strategy (0-4)"
    )

    # Temporal features
    tx_hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    tx_day: int = Field(..., ge=1, le=31, description="Day of month")
    tx_month: int = Field(..., ge=1, le=12, description="Month")
    tx_year: int = Field(..., ge=2018, le=2030, description="Year")
    tx_day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Mon)"
    )

    # Debit flag
    is_debit: int = Field(
        ..., ge=0, le=1, description="1=debit transaction, 0=credit"
    )

    # Customer aggregate features
    cust_total_value: float = Field(
        ..., description="Customer total transaction value"
    )
    cust_avg_value: float = Field(
        ..., description="Customer average transaction value"
    )
    cust_tx_count: int = Field(
        ..., ge=1, description="Customer transaction count"
    )
    cust_std_value: float = Field(
        ..., ge=0, description="Std dev of customer transaction values"
    )
    cust_recency_days: float = Field(
        ..., ge=0, description="Days since last transaction"
    )
    cust_max_value: float = Field(
        ..., description="Customer max single transaction value"
    )
    cust_min_value: float = Field(
        ..., description="Customer min single transaction value"
    )
    cust_median_value: float = Field(
        ..., description="Customer median transaction value"
    )
    cust_unique_products: int = Field(
        ..., ge=1, description="Unique products used by customer"
    )

    # Categorical encoded features (one-hot / label encoded)
    ProductCategory_airtime: Optional[float] = 0.0
    ProductCategory_data_bundles: Optional[float] = 0.0
    ProductCategory_financial_services: Optional[float] = 0.0
    ProductCategory_movies: Optional[float] = 0.0
    ProductCategory_other: Optional[float] = 0.0
    ProductCategory_ticket: Optional[float] = 0.0
    ProductCategory_transport: Optional[float] = 0.0
    ProductCategory_tv: Optional[float] = 0.0
    ProductCategory_utility_bill: Optional[float] = 0.0
    ChannelId_ChannelId_2: Optional[float] = 0.0
    ChannelId_ChannelId_3: Optional[float] = 0.0
    ChannelId_ChannelId_5: Optional[float] = 0.0
    ProviderId_ProviderId_2: Optional[float] = 0.0
    ProviderId_ProviderId_3: Optional[float] = 0.0
    ProviderId_ProviderId_4: Optional[float] = 0.0
    ProviderId_ProviderId_5: Optional[float] = 0.0
    ProviderId_ProviderId_6: Optional[float] = 0.0
    ProductId: Optional[float] = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Value": 5000.0,
                "PricingStrategy": 2,
                "tx_hour": 14,
                "tx_day": 15,
                "tx_month": 11,
                "tx_year": 2018,
                "tx_day_of_week": 3,
                "is_debit": 1,
                "cust_total_value": 45000.0,
                "cust_avg_value": 4500.0,
                "cust_tx_count": 10,
                "cust_std_value": 1200.0,
                "cust_recency_days": 5.0,
                "cust_max_value": 12000.0,
                "cust_min_value": 500.0,
                "cust_median_value": 4000.0,
                "cust_unique_products": 3,
                "ProductCategory_airtime": 1.0,
                "ChannelId_ChannelId_2": 1.0,
                "ProviderId_ProviderId_4": 1.0,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being high-risk (0=low risk, 1=high risk)",
    )
    risk_label: str = Field(
        ..., description="Human-readable risk label: 'high_risk' or 'low_risk'"
    )
    threshold_used: float = Field(
        ..., description="Decision threshold applied"
    )
    model_version: str = Field(..., description="MLflow model version used")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="List of customer feature sets (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_name: str
    model_version: str
    mlflow_tracking_uri: str
