"""
Data split utilities for time-series datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from toolkits.logger import get_logger
from tqdm import tqdm


@dataclass(frozen=True)
class TrainTestAtDate:
    """One split at anchor date ``as_of``."""

    as_of: pd.Timestamp
    X_train: pd.DataFrame
    y_train: pd.Series | pd.DataFrame | None
    X_test: pd.DataFrame
    y_test: pd.Series | pd.DataFrame | None


class Data(ABC):
    """
    Pure dataset split base class.

    Notes:
    - ``load_target`` is required.
    - ``load_features`` is optional; if ``feature_col=[]`` then X comes from target's date only.
    - Base class does NOT align target/features by date; derived class must ensure alignment.
    - ``future_period_used = k`` means target at row T depends on future T+1..T+k.
    """

    def __init__(
        self,
        date_col: str,
        *,
        target_col: str = "y",
        feature_col: list[str] | None = None,
        future_period_used: int = 0,
        rolling_window: int = 250,
        m: int = 1,
        name: str = "Data",
        logger: logging.Logger | None = None,
    ) -> None:
        if not (isinstance(date_col, str) and date_col.strip()):
            raise ValueError("date_col must be a non-empty string.")
        if feature_col is None:
            feature_col = []
        if rolling_window <= 0:
            raise ValueError("rolling_window must be > 0.")
        if m <= 0:
            raise ValueError("m must be > 0.")
        if future_period_used < 0:
            raise ValueError("future_period_used must be >= 0.")

        self.date_col = date_col
        self.target_col = target_col
        self.feature_col = feature_col
        self.future_period_used = int(future_period_used)
        self.rolling_window = int(rolling_window)
        self.m = int(m)
        self.name = name
        self.logger = logger if logger is not None else get_logger(name)

        self._prepared = False
        self.target = self.load_target()
        self.features = self.load_features() if self.feature_col else None

    @abstractmethod
    def load_target(self) -> pd.DataFrame:
        """Return full target frame. Must include ``date_col`` and ``target_col``."""
        raise NotImplementedError

    def load_features(self) -> pd.DataFrame:
        """Return full feature frame. Must already align with target by ``date_col``."""
        raise NotImplementedError(
            "Implement load_features when feature_col is non-empty."
        )

    def prepare(self) -> None:
        """Normalize date types, build future-date availability, and basic checks."""
        if self._prepared:
            return

        if self.date_col not in self.target.columns:
            raise KeyError(f"target missing date_col: {self.date_col}")
        if self.target_col not in self.target.columns:
            raise KeyError(f"target missing target_col: {self.target_col}")

        self.target = self.target.copy()
        self.target[self.date_col] = pd.to_datetime(self.target[self.date_col])
        self.target = self.target.sort_values(self.date_col).reset_index(drop=True)
        self.target["future_date"] = self.target[self.date_col].shift(
            -self.future_period_used
        )

        if self.features is None:
            self.features = self.target[[self.date_col, "future_date"]].copy()
        else:
            self.features = self.features.copy()
            if self.date_col not in self.features.columns:
                raise KeyError(f"features missing date_col: {self.date_col}")
            for col in self.feature_col:
                if col not in self.features.columns:
                    raise KeyError(f"features missing feature_col: {col}")
            self.features[self.date_col] = pd.to_datetime(self.features[self.date_col])
            keep_cols = [self.date_col, *self.feature_col]
            self.features = self.features[keep_cols].sort_values(self.date_col).reset_index(drop=True)
            # Do not realign/merge target and features; derived class is responsible for alignment.
            self.features["future_date"] = self.features[self.date_col].shift(
                -self.future_period_used
            )

        self._prepared = True

    def _prediction_target_date_map(self) -> pd.Series:
        """For each ``date_col`` value, calendar date when ``target_col`` is realized (k rows ahead)."""
        t = self.target[[self.date_col]].sort_values(self.date_col).reset_index(drop=True)
        d = pd.to_datetime(t[self.date_col])
        ptd = d.shift(-self.future_period_used)
        return pd.Series(ptd.values, index=d.values)

    def _attach_prediction_target_date(self, x: pd.DataFrame) -> pd.DataFrame:
        if len(x) == 0:
            return x
        out = x.copy()
        m = self._prediction_target_date_map()
        out["prediction_target_date"] = pd.to_datetime(out[self.date_col]).map(m)
        return out

    def split(
        self, date: str | pd.Timestamp | datetime, *, include_targets: bool = True
    ) -> TrainTestAtDate:
        """
        Split at anchor date D.

        - train: latest ``rolling_window`` rows with label already observable at D
        - test: first ``m`` rows with ``date_col >= D``
        - ``X_*`` include ``date_col``, all ``feature_col``, and ``prediction_target_date``
          (calendar day when ``target_col`` is realized; ``k`` rows ahead in sorted target).
        - ``y_*`` are DataFrames: ``date_col``, ``target_col``, ``prediction_target_date``.
        """
        self.prepare()
        anchor = pd.to_datetime(date)

        if self.future_period_used == 0:
            x_train_mask = self.features[self.date_col] < anchor
            y_train_mask = self.target[self.date_col] < anchor
        else:
            x_train_mask = self.features["future_date"] <= anchor
            y_train_mask = self.target["future_date"] <= anchor

        X_train = self.features.loc[x_train_mask].tail(self.rolling_window).reset_index(drop=True)
        X_test = self.features.loc[self.features[self.date_col] >= anchor].head(self.m).reset_index(drop=True)
        X_train = X_train.drop(columns=["future_date"], errors="ignore")
        X_test = X_test.drop(columns=["future_date"], errors="ignore")
        X_train = self._attach_prediction_target_date(X_train)
        X_test = self._attach_prediction_target_date(X_test)

        if not include_targets:
            return TrainTestAtDate(anchor, X_train, None, X_test, None)

        y_train_df = self.target.loc[y_train_mask].tail(self.rolling_window).reset_index(drop=True)
        y_test_df = self.target.loc[self.target[self.date_col] >= anchor].head(self.m).reset_index(drop=True)

        ptd_map = self._prediction_target_date_map()
        if self.target_col in y_train_df.columns:
            y_train = y_train_df[[self.date_col, self.target_col]].copy()
            y_train[self.target_col] = pd.to_numeric(y_train[self.target_col], errors="coerce")
            y_train["prediction_target_date"] = pd.to_datetime(y_train[self.date_col]).map(ptd_map)
        else:
            y_train = pd.DataFrame(columns=[self.date_col, self.target_col, "prediction_target_date"])

        if self.target_col in y_test_df.columns:
            y_test = y_test_df[[self.date_col, self.target_col]].copy()
            y_test[self.target_col] = pd.to_numeric(y_test[self.target_col], errors="coerce")
            y_test["prediction_target_date"] = pd.to_datetime(y_test[self.date_col]).map(ptd_map)
        else:
            y_test = pd.DataFrame(columns=[self.date_col, self.target_col, "prediction_target_date"])

        return TrainTestAtDate(anchor, X_train, y_train, X_test, y_test)

    def plot_feature_importance(self):
        """
        Train a RandomForest on full history and plot feature importance.

        Returns:
            tuple[pd.DataFrame, plt.Figure]:
                - importance table sorted desc (feature, importance)
                - matplotlib figure
        """
        self.prepare()
        if self.features is None:
            raise ValueError("No features found. Set feature_col and implement load_features().")

        # Build full-history supervised table by date.
        x_df = self.features.copy()
        y_df = self.target[[self.date_col, self.target_col]].copy()
        x_df[self.date_col] = pd.to_datetime(x_df[self.date_col], errors="coerce")
        y_df[self.date_col] = pd.to_datetime(y_df[self.date_col], errors="coerce")
        data = x_df.merge(y_df, on=self.date_col, how="inner")

        # Remove meta/time columns from feature matrix.
        drop_cols = [self.date_col, "future_date", self.target_col]
        feat_cols = [c for c in data.columns if c not in drop_cols]
        if len(feat_cols) == 0:
            raise ValueError("No usable feature columns after excluding date/future_date/target.")

        X = data[feat_cols].copy()
        y = pd.to_numeric(data[self.target_col], errors="coerce")
        valid = y.notna()
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)
        if len(X) < 20:
            raise ValueError("Not enough valid rows to train RandomForest for importance.")

        # Encode categoricals if present.
        X = pd.get_dummies(X, drop_first=False)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        imp_df = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(imp_df))))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#2a6f97")
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance (RandomForest) - {self.name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return imp_df, fig
