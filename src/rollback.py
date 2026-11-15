
from pathlib import Path
import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

client = MlflowClient()
MODEL_NAME = "CustomerChurnModel"

# Show available models
models = client.search_registered_models()

for model in models:
    print("Registered model:", model.name)

def rollback(version):
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    available = [v.version for v in versions]

    if str(version) not in available:
        print("Version not found!")
        print("Available versions:", available)
        return

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version
    )

    print(f"Rollback successful: Version {version}")


if __name__ == "__main__":
    rollback(2)




# from pathlib import Path
# import mlflow
# from mlflow import MlflowClient

# ROOT = Path(__file__).resolve().parent.parent

# mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

# MODEL_NAME = "CustomerChurnModel"

# client = MlflowClient()


# def rollback(version):
#     client.set_registered_model_alias(
#         name=MODEL_NAME,
#         alias="champion",
#         version=version
#     )

#     print(f"Champion changed to version {version}")


# if __name__ == "__main__":
#     rollback(2)
