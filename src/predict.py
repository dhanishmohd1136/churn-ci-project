
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


ROOT = Path(__file__).resolve().parent.parent

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

MODEL_NAME = "CustomerChurnModel"

app = FastAPI(title="Customer Churn Prediction API")


# Input validation
class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(ge=18, le=100)
    Gender: Literal["Male", "Female"]
    Tenure: int = Field(ge=0)

    Usage_Frequency: int = Field(
        ge=0, alias="Usage Frequency"
    )
    Support_Calls: int = Field(
        ge=0, alias="Support Calls"
    )
    Payment_Delay: int = Field(
        ge=0, alias="Payment Delay"
    )

    Subscription_Type: Literal[
        "Basic", "Standard", "Premium"
    ] = Field(alias="Subscription Type")

    Contract_Length: Literal[
        "Monthly", "Quarterly", "Annual"
    ] = Field(alias="Contract Length")

    Total_Spend: float = Field(
        ge=0, alias="Total Spend"
    )
    Last_Interaction: int = Field(
        ge=0, alias="Last Interaction"
    )


# Output validation
class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    churn: bool


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):

    try:
        # Load current champion model
        model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        # Convert input to DataFrame
        data = pd.DataFrame([
            customer.model_dump(by_alias=True)
        ])

        # Predict
        prediction = int(model.predict(data)[0])

        return PredictionOutput(
            prediction=prediction,
            churn=bool(prediction)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed"
        ) from e
