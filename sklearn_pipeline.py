import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class AI4IFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.iloc[:, 2:-5]

        X.columns = X.columns.str.replace(r"\s*\[.*?\]", "", regex=True)

        X = pd.get_dummies(X, columns=["Type"], drop_first=True)

        X["temp_diff"] = X["Process temperature"] - X["Air temperature"]
        X["torque_speed_ratio"] = X["Torque"] / X["Rotational speed"]
        X["wear_rate"] = X["Tool wear"] / X["Rotational speed"]

        return X


fe_pipeline = Pipeline([
    ("feature_engineering", AI4IFeatureEngineering())
])