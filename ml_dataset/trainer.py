from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from toolkits.ml_dataset import Data
from tqdm import tqdm
import matplotlib.pyplot as plt


# Jiuzhaigou-style daily caps (aligned to the **prediction target** calendar date).
_PEAK_MAX = 41_000  # 旺季 4月1日—11月15日
_OFF_MAX = 23_000  # 淡季 11月16日—次年3月31日


def daily_visitor_capacity_cap(prediction_date: pd.Timestamp) -> int:
    """
    Return max allowed visitors for ``prediction_date`` (month/day, ignores year).

    Peak: Apr 1 through Nov 15. Off season: Nov 16 through Mar 31.
    """
    d = pd.to_datetime(prediction_date, errors="coerce")
    if pd.isna(d):
        raise ValueError("prediction_date is NaT or unparseable.")
    m, day = int(d.month), int(d.day)
    if m in (5, 6, 7, 8, 9, 10):
        return _PEAK_MAX
    if m == 4:
        return _PEAK_MAX
    if m == 11:
        return _PEAK_MAX if day <= 15 else _OFF_MAX
    # Dec, Jan, Feb, Mar
    return _OFF_MAX


def clip_prediction_to_capacity(pred: float, prediction_date: pd.Timestamp) -> float:
    """Upper-bound ``pred`` by season capacity; NaN passes through."""
    v = pd.to_numeric(pred, errors="coerce")
    if pd.isna(v):
        return float("nan")
    d = pd.to_datetime(prediction_date, errors="coerce")
    if pd.isna(d):
        # No valid evaluation date — return uncapped prediction (caller should skip).
        return float(v)
    cap = daily_visitor_capacity_cap(d)
    return min(float(v), float(cap))


class Trainer(ABC):
    """
    Stateless w.r.t. train/test matrices: pass ``X_train, y_train, X_test, y_test``
    into ``train_predict`` each time. Subclasses may use ``__init__`` only for
    hyperparameters (e.g. ``adj``).
    """

    @abstractmethod
    def train_predict(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, pd.DataFrame],
    ) -> tuple:
        """
        Train (if needed) and predict on the given split.

        Returns:
            (out_date, pred, actual) — ``out_date`` is the label / evaluation date
            (e.g. ``prediction_target_date`` for the first test row).
        """
        raise NotImplementedError


class TrainHistory:
    def __init__(self, start, data: Data, trainer: Trainer, min_window: int):
        self.start = pd.to_datetime(start)
        self.data = data
        self.trainer = trainer
        self.min_window = min_window
        self.preds: list = []
        self.actuals: list = []
        self.dates: list = []
        self.results: pd.DataFrame | None = None

    def run(self):
        self.data.prepare()
        self.preds = []
        self.actuals = []
        self.dates = []
        self.results = None

        dates = self.data.target[
            self.data.target[self.data.date_col] >= self.start
        ][self.data.date_col]
        for date in tqdm(dates):
            split = self.data.split(date)
            if len(split.X_train) < self.min_window or len(split.X_test) == 0:
                continue
            out_date, pred, actu = self.trainer.train_predict(
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
            )
            out_dt = pd.to_datetime(out_date, errors="coerce")
            if not pd.notna(out_dt):
                # No realized prediction date (e.g. horizon exceeds series end).
                continue
            pred = clip_prediction_to_capacity(pred, out_dt)
            self.preds.append(pred)
            self.actuals.append(actu)
            self.dates.append(pd.to_datetime(out_date))
        return self

    def evaluate(self, plot: bool = True, save_path: str = None):
        res = pd.DataFrame(
            {"date": self.dates, "pred": self.preds, "actual": self.actuals}
        )
        if len(res) == 0:
            raise ValueError("No prediction records. Run `run()` first.")
        res["date"] = pd.to_datetime(res["date"])
        res["diff"] = res["pred"] - res["actual"]
        res = res.sort_values("date").reset_index(drop=True)
        self.results = res

        if plot:
            self._plot_timeseries(res, save_path)
        rmse = self._cal_rmse(res)
        return res, rmse

    def plot_timeseries(self, save_path: str = None):
        """Plot ``self.results`` (call ``evaluate(plot=False)`` first if needed)."""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to plot. Run run() then evaluate(plot=False).")
        self._plot_timeseries(self.results, save_path)

    def _plot_timeseries(self, res: pd.DataFrame, save_path: str = None):
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        ax.plot(res["date"], res["actual"], label="实际值", linewidth=1, linestyle="--",color="#d62728")
        ax.plot(
            res["date"],
            res["pred"],
            label="预测值",
            linewidth=1,
            color="#1f77b4",
            alpha=0.9,
        )
        # ax.set_title("Train History: prediction vs actual")
        ax.set_xlabel("日期")
        ax.set_ylabel("客流人数")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig

    def _cal_rmse(self, res: pd.DataFrame):
        diff = pd.to_numeric(res["pred"], errors="coerce") - pd.to_numeric(
            res["actual"], errors="coerce"
        )
        rmse = (diff.pow(2).mean()) ** 0.5
        return float(rmse)
