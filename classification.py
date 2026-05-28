# =========================
# IMPORTS
# =========================
import polars as pl
import s3fs
import numpy as np
from dotenv import load_dotenv

from torchTextClassifiers import torchTextClassifiers

# =========================
# SETUP
# =========================
load_dotenv(override=True)

# =========================
# LOAD DATA (til mapping code -> label)
# =========================
df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

# Lav hurtig mapping (meget hurtigere end filter)
code_to_label = dict(zip(df["code"], df["name"]))

# =========================
# HELPERS
# =========================
def get_label_from_code(code):
    return code_to_label.get(code, "Ukendt")


def predict_text(text, model, threshold=0.90):
    """
    Predict branche (code + label) fra tekst.
    """

    X = np.array([[text]])

    result = model.predict(X, top_k=1)

    pred_code = result["prediction"][0][0]
    confidence = result["confidence"][0][0].item()

    if confidence < threshold:
        return {
            "label": "Ukendt - giv mere information",
            "code": None,
            "confidence": round(confidence, 3),
        }

    label = get_label_from_code(pred_code)

    return {
        "label": label,
        "code": pred_code,
        "confidence": round(confidence, 3),
    }

# =========================
# LOAD MODEL (fra S3/MLflow artifacts)
# =========================
fs = s3fs.S3FileSystem(
    anon=True,
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"

fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)

# Load model
ttc = torchTextClassifiers.load(local_dir)
ttc.pytorch_model.eval()

# =========================
# TEST / EKSEMPEL
# =========================


result = predict_text("Sale of dairy", ttc)
print("Output:", result)

result = predict_text("Goods retail", ttc)
print("Output:", result)