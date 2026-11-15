
import pandas as pd
from src.train import train_model

def test_training(tmp_path):
    data = pd.DataFrame({
        "CustomerID": range(1, 21),
        "Age": [25, 30] * 10,
        "Gender": ["Male", "Female"] * 10,
        "Tenure": [12, 24] * 10,
        "Churn": [0, 1] * 10
    })

    file = tmp_path / "test.csv"
    data.to_csv(file, index=False)

    model, X_train, metrics = train_model(file)

    assert model is not None
    assert len(X_train) > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
