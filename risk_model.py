import argparse
import joblib
import pandas as pd
from sklearn_pipeline import AI4IFeatureEngineering

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

def load_data(data_path):
    return pd.read_csv(data_path)

def prepare_features_and_target(df): 
    fe_transformer = AI4IFeatureEngineering()
    y = df["Machine failure"].copy()
    X = fe_transformer.fit_transform(df)
    return X, y, fe_transformer

def train_and_evaluate(
    data_path: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    model_out: str,
    features_out: str,
) -> None:
    X, y, _ = prepare_features_and_target(load_data(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # -------------------
    # Model
    # -------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss"
    )

    # -------------------
    # Cross-validation on train only
    # -------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision"
        },
        n_jobs=-1
    )

    print("=== Cross-validation (train) ===")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric:10s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # -------------------
    # Final training
    # -------------------
    model.fit(X_train, y_train)

    # -------------------
    # Test evaluation
    # -------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Test set ===")
    print(f"ROC AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR AUC  : {average_precision_score(y_test, y_proba):.4f}")

    print("\nConfusion matrix :")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report :")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------
    # Feature importance
    # -------------------
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n=== Feature importance ===")
    print(feat_imp)

    # -------------------
    # Save model + columns
    # -------------------
    joblib.dump(model, model_out)
    joblib.dump(list(X_train.columns), features_out)

    print(f"\nModèle sauvegardé dans {model_out}")
    print(f"Colonnes sauvegardées dans {features_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle XGBoost pour prédire les pannes machine."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/ai4i2020.csv",
        help="Chemin vers le fichier CSV de données."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed aléatoire."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Nombre d'arbres XGBoost."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Profondeur maximale des arbres."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Taux d'apprentissage."
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="xgb_model.pkl",
        help="Chemin de sortie pour sauvegarder le modèle."
    )
    parser.add_argument(
        "--features-out",
        type=str,
        default="xgb_features.pkl",
        help="Chemin de sortie pour sauvegarder les colonnes utilisées."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_and_evaluate(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        model_out=args.model_out,
        features_out=args.features_out,
    )