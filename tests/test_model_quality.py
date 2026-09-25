import pandas as pd
import mlflow
from sklearn.metrics import recall_score

RECALL_THRESHOLD = 0.60
MODEL_PATH = "model"
REFERENCE_DATA = "data/reference/diabetes_reference.csv"

FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]


def test_recall_above_threshold():
    model = mlflow.sklearn.load_model(MODEL_PATH)

    df = pd.read_csv(REFERENCE_DATA)
    X_ref = df[FEATURES]
    y_ref = df["outcome"]

    y_pred = model.predict(X_ref)
    recall = recall_score(y_ref, y_pred)

    print(f"Rappel sur le jeu de reference : {recall:.4f}")

    assert recall >= RECALL_THRESHOLD, (
        f"Rappel {recall:.4f} sous le seuil requis de {RECALL_THRESHOLD}"
    )