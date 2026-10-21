"""
model.py  --  Jury 2 shared model code
House Price Regression (Ames Housing)

Used by BOTH the Colab notebook and the Hugging Face app, so the numbers you
show in the jury and the predictions the frontend serves come from the same code.

Contents
--------
- AmesFeatureEngineer : leakage-safe, domain-aware cleaning + feature engineering
- build_preprocessor  : one-hot (rare categories grouped) + scaling
- build_pipeline      : FE -> preprocessing -> model (optionally with log-target)
- build_jury1_pipeline: the exact Jury 1 baseline pipeline (for fair comparison)
- remove_outliers     : drops the documented Ames outliers (training data only)
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "SalePrice"

# In Ames, a missing value in these columns means "the house does not have this
# feature" (no alley, no basement, no garage, no pool ...), NOT "data was lost".
NONE_MEANS_ABSENT = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Pool QC", "Fence", "Misc Feature",
    "Mas Vnr Type",
]

# Numeric columns where missing also means "absent" -> 0
ZERO_IF_MISSING = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
]

# Quality ratings are ordered (Ex > Gd > TA > Fa > Po) -> give them an ordinal copy
QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUAL_COLS = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
    "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC",
]


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-aware cleaning + feature engineering.

    Anything that is *learned from data* (neighbourhood medians, the most common
    electrical system) is learned in fit() on TRAINING data only, then reused in
    transform() -> no data leakage, and safe inside a scikit-learn Pipeline.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.lot_by_nbhd_ = X.groupby("Neighborhood")["Lot Frontage"].median().to_dict()
        self.lot_global_ = float(X["Lot Frontage"].median())
        self.electrical_mode_ = X["Electrical"].mode()[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # ---- 1. domain-aware missing values ---------------------------------
        for c in NONE_MEANS_ABSENT:
            if c in X.columns:
                X[c] = X[c].fillna("None")
        if "Electrical" in X.columns:
            X["Electrical"] = X["Electrical"].fillna(self.electrical_mode_)
        # Lot frontage: typical value for the house's own neighbourhood
        X["Lot Frontage"] = (
            X["Lot Frontage"]
            .fillna(X["Neighborhood"].map(self.lot_by_nbhd_))
            .fillna(self.lot_global_)
        )
        # No garage -> garage "built" when the house was
        X["Garage Yr Blt"] = X["Garage Yr Blt"].fillna(X["Year Built"])
        for c in ZERO_IF_MISSING:
            if c in X.columns:
                X[c] = X[c].fillna(0)

        # MS SubClass is a category code, not a quantity
        X["MS SubClass"] = X["MS SubClass"].astype(int).astype(str)

        # ---- 2. engineered features -----------------------------------------
        X["TotalSF"] = X["Total Bsmt SF"] + X["1st Flr SF"] + X["2nd Flr SF"]
        X["TotalBath"] = (
            X["Full Bath"] + 0.5 * X["Half Bath"]
            + X["Bsmt Full Bath"] + 0.5 * X["Bsmt Half Bath"]
        )
        X["HouseAge"] = (X["Yr Sold"] - X["Year Built"]).clip(lower=0)
        X["RemodAge"] = (X["Yr Sold"] - X["Year Remod/Add"]).clip(lower=0)
        X["PorchDeckSF"] = (
            X["Wood Deck SF"] + X["Open Porch SF"] + X["Enclosed Porch"]
            + X["3Ssn Porch"] + X["Screen Porch"]
        )
        X["HasPool"] = (X["Pool Area"] > 0).astype(int)
        X["HasGarage"] = (X["Garage Area"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["HasBasement"] = (X["Total Bsmt SF"] > 0).astype(int)
        X["IsRemodeled"] = (X["Year Remod/Add"] != X["Year Built"]).astype(int)
        X["QualxArea"] = X["Overall Qual"] * X["Gr Liv Area"]   # quality x size interaction
        X["LogGrLivArea"] = np.log1p(X["Gr Liv Area"])
        X["LogLotArea"] = np.log1p(X["Lot Area"])

        # ordinal copies of the quality ratings
        for c in QUAL_COLS:
            if c in X.columns:
                X[c + " Ord"] = X[c].map(QUAL_MAP).fillna(0)

        return X


def build_preprocessor():
    """Scale numerics; one-hot categoricals. Categories seen fewer than 10 times
    are grouped into one 'infrequent' bucket (fixes the unstable rare-category
    coefficients we found in Jury 1)."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def build_pipeline(model, feature_engineering=True, log_target=True):
    """Jury 2 pipeline: [FE] -> preprocessing -> model, optionally trained on log(price)."""
    steps = []
    if feature_engineering:
        steps.append(("fe", AmesFeatureEngineer()))
    steps += [("prep", build_preprocessor()), ("model", model)]
    pipe = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def build_jury1_pipeline(model, numerical_features, categorical_features):
    """The exact Jury 1 preprocessing (median / most-frequent imputation + one-hot)."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric, numerical_features),
        ("cat", categorical, categorical_features),
    ])
    return Pipeline([("preprocessor", pre), ("model", model)])


def remove_outliers(X, y, area_threshold=4000):
    """Drop the documented Ames outliers (huge, very cheap 'Partial' sales).
    Applied to TRAINING data only; the test set is never touched."""
    keep = X["Gr Liv Area"] <= area_threshold
    return X[keep], y[keep]


# Best hyper-parameters found by RandomizedSearchCV in the notebook (Section 9).
# Stored here so the Hugging Face app can rebuild the same model without re-searching.
BEST_GB_PARAMS = dict(
    n_estimators=500, learning_rate=0.08, max_depth=3, subsample=1.0,
    min_samples_leaf=3, max_features=0.5, random_state=RANDOM_STATE,
)

# Model chosen by 5-fold cross-validation on the training data (notebook Section 8):
# Ridge and Lasso were statistically tied (CV RMSE ~19.0K); Ridge has the lower variance.
# "ridge" or "gb" -> which model the app uses as its headline prediction.
FINAL_MODEL_KEY = "ridge"
