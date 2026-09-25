import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from mlflow import MlflowClient
import shutil
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
USE_REGISTRY = os.getenv("USE_REGISTRY", "true").lower() == "true"
mlflow.set_experiment("diabetes-risk")

df = pd.read_csv("data/raw/diabetes_train.csv")

FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]

X = df[FEATURES]
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("undersampler", RandomUnderSampler(random_state=42)),
    ("model", RandomForestClassifier(random_state=42)),
])

param_grid = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_leaf": [1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(pipeline, param_grid=param_grid, scoring="recall", cv=cv, n_jobs=-1)

with mlflow.start_run(run_name="entrainement-rf"):

    search.fit(X_train, y_train)

    for i, (params, score) in enumerate(zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
    )):
        with mlflow.start_run(run_name=f"combo-{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_recall", score)

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("cv_best_recall", search.best_score_)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
    mlflow.log_metric("test_auc", roc_auc_score(y_test, y_proba))

    print("recall:", round(recall_score(y_test, y_pred), 4))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("false_negatives", fn)
    mlflow.log_metric("true_positives", tp)

    shutil.rmtree("model", ignore_errors=True)
    mlflow.sklearn.save_model(
        sk_model=best_pipeline,
        path="model",
        input_example=X_test.head(),
        skops_trusted_types=[
            "imblearn.pipeline.Pipeline",
            "imblearn.under_sampling._prototype_selection._random_under_sampler.RandomUnderSampler",
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
        ],
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="model",
        registered_model_name="diabetes-risk-model" if USE_REGISTRY else None,
        input_example=X_test.head(),
        skops_trusted_types=[
            "imblearn.pipeline.Pipeline",
            "imblearn.under_sampling._prototype_selection._random_under_sampler.RandomUnderSampler",
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
        ],
    )


    if USE_REGISTRY:
        client = MlflowClient()
        version = client.get_latest_versions("diabetes-risk-model")[0].version

        client.set_registered_model_alias(
            name="diabetes-risk-model",
            alias="production",
            version=version,
        )
        client.transition_model_version_stage(
            name="diabetes-risk-model",
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )