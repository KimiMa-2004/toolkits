'''
Author: Qimin Ma
Date: 2026-04-04 16:05:42
LastEditTime: 2026-04-05 21:48:40
FilePath: /Toolkit/toolkit/ml_dataset/ts_split.py
Description:
    Rolling train / predict on a time index with optional features.
    Label timing uses ``future_date`` so training rows only include targets
    already realized by the as-of date (reduces lookahead leakage).
Copyright (c) 2026 by Qimin Ma, All Rights Reserved.
'''
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from abc import ABC, abstractmethod

from toolkit.logger import get_logger

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:

    def _tqdm(iterable, **kwargs):
        return iterable


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


class ts_dataset_split(ABC):
    """
    Walk-forward rolling fit/predict on a time-ordered target (optional exogenous features).

    **Label timing:** ``future_date`` is the calendar index at which the target for a row
    becomes observable. Training at ``anchor`` only uses rows with
    ``future_date <= anchor`` (or ``date < anchor`` when ``future_period_used == 0``),
    so the model does not train on labels that are still unknown at ``anchor``.

    **Target-only:** leave ``feature_col=[]``; ``features`` is then just dates plus
    ``future_date`` (suitable for ARIMA / univariate models that read ``y_train`` only).

    **``future_period_used = k`` (horizon in *trading rows*, not calendar days):**
    Row ``date_col == T`` holds a target ``y`` that is only fully known after the next
    ``k`` bars in your series (e.g. after T+1…T+k closes). Examples: k=1 next-day
    return; k=10 cumulative return or realized volatility over T+1…T+k — you define
    ``y``; the pipeline only needs ``k`` so that ``future_date`` lines up with when
    that label becomes observable.

    Preprocessing sets ``future_date = date_col.shift(-k)`` (the calendar stamp of the
    **k-th** forward row), and training at anchor ``D`` keeps rows with
    ``future_date <= D`` so no row is used whose label is still unknown at ``D``.
    When ``k == 0``, ``y`` is treated as known on the same bar and train uses
    ``date_col < anchor`` to avoid same-bar leakage.
    """

    def __init__(self, date_col:str='TradingDay', \
                        feature_col:list[str]=[], \
                        target_col:str="y", \
                        mode:str="pd", \
                        future_period_used:int=0, \
                        window_size:int=250,
                        rolling_freq:int=1,
                        name:str="Model",
                        logger:logging.Logger=None,
                        walkforward_debug:bool=False):
        # future_period_used=k: target on row T depends on (or is realized after) the
        # next k rows in date order (e.g. T+1…T+k return sum, or vol over that window).
        # Must match how you construct ``target_col`` vs ``date_col``.
        self.name = name
        if logger is None:
            self.logger = get_logger(self.name)
        else:
            self.logger = logger

        self.logger.info(f"Initializing {self.name}...")
        
        self.future_period_used = future_period_used
        self.window_size = window_size
        self.date_col = date_col
        self.feature_col = feature_col
        self.target_col = target_col
        self.mode = mode
        self.rolling_freq = rolling_freq
        self.walkforward_debug = walkforward_debug

        if len(self.feature_col) == 0:
            self.with_features = False
        else:
            self.with_features = True

        self.target = self._load_target()
        if self.with_features:
            self.features = self._load_features()
        else:
            self.features = None

    def _load_features(self) -> pd.DataFrame | pl.DataFrame | str:
        """Override when ``feature_col`` is non-empty; otherwise not used."""
        raise NotImplementedError(
            "Implement _load_features when using non-empty feature_col "
            "(e.g. external regressors). Target-only models can leave feature_col=[]."
        )

    @abstractmethod
    def _load_target(self) -> pd.DataFrame | pl.DataFrame | str:
        pass


    def _data_preprocessing(self):
        if self.mode == "pd":
            self.target[self.date_col] = pd.to_datetime(self.target[self.date_col])
            self.target = self.target[[self.date_col, self.target_col]]
            self.target.sort_values(by=self.date_col, inplace=True)
            # Row T: future_date = trading date k steps ahead → label y_T is admissible in
            # training at anchor D only when future_date_T <= D.
            self.target["future_date"] = self.target[self.date_col].shift(
                -self.future_period_used
            )

            if self.with_features:
                self.features[self.date_col] = pd.to_datetime(
                    self.features[self.date_col]
                )
                self.features = self.features[[self.date_col, *self.feature_col]]
                self.features.sort_values(by=self.date_col, inplace=True)
                self.features = self.features.merge(
                    self.target[[self.date_col, "future_date"]],
                    on=self.date_col,
                    how="inner",
                )
            else:
                # Univariate path (ARIMA, etc.): features are calendar + label timing only.
                self.features = self.target[[self.date_col, "future_date"]].copy()

            n_na = self.target["future_date"].isna().sum()
            if n_na:
                self.logger.debug(
                    "Last {} target rows have NaN future_date (horizon past series end).",
                    int(n_na),
                )

            self.logger.info(
                "Preprocessed: {} target rows, {} feature rows (with_features={}, "
                "future_period_used={}).",
                len(self.target),
                len(self.features),
                self.with_features,
                self.future_period_used,
            )
        elif self.mode == "pl":
            raise NotImplementedError(
                "mode='pl' is not supported with the current leakage-safe preprocessing; "
                "use mode='pd'."
            )

        self._additional_data_preprocessing()
            
            
    @abstractmethod
    def _additional_data_preprocessing(self):
        pass


    def training_target_history(self, anchor) -> pd.DataFrame:
        """Training-window targets used at ``anchor``: columns ``date``, ``y``."""
        if self.mode != "pd":
            raise NotImplementedError("training_target_history is implemented for mode='pd' only.")
        _, y_train, _, _ = self._split_data(anchor)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(y_train[self.date_col]),
                "y": y_train[self.target_col].astype(float),
            }
        )

    def _split_data(self, anchor):
        """Train on past realized labels; test rows are at calendar date >= anchor."""
        if self.mode == "pd":
            # k>0: label for row d is realized on future_date(d); include while future_date<=anchor.
            # k==0: future_date equals date; use strict date < anchor so the test bar is not in train.
            if self.future_period_used == 0:
                mask_train = self.features[self.date_col] < anchor
                y_mask = self.target[self.date_col] < anchor
            else:
                mask_train = self.features["future_date"] <= anchor
                y_mask = self.target["future_date"] <= anchor
            X_train = self.features.loc[mask_train].tail(self.window_size)
            y_train = self.target.loc[y_mask].tail(self.window_size)
            mask_test = self.features[self.date_col] >= anchor
            X_test = self.features.loc[mask_test].head(self.rolling_freq)
            y_test = self.target.loc[self.target[self.date_col] >= anchor].head(
                self.rolling_freq
            )
            return X_train, y_train, X_test, y_test

        elif self.mode == "pl":
            if self.future_period_used == 0:
                fx_m = pl.col(self.date_col) < anchor
                y_m = pl.col(self.date_col) < anchor
            else:
                fx_m = pl.col("future_date") <= anchor
                y_m = pl.col("future_date") <= anchor
            X_train = self.features.filter(fx_m).tail(self.window_size)
            y_train = self.target.filter(y_m).tail(self.window_size)
            X_test = self.features.filter(pl.col(self.date_col) >= anchor).head(
                self.rolling_freq
            )
            y_test = self.target.filter(pl.col(self.date_col) >= anchor).head(
                self.rolling_freq
            )
            return X_train, y_train, X_test, y_test

        else:
            pass

    @abstractmethod
    def _train_function(self, \
        X:pd.DataFrame | pl.DataFrame, \
        y:pd.DataFrame | pl.DataFrame, \
        model_path:str=None):
        pass


    @abstractmethod
    def _predict_function(self, \
        X_test:pd.DataFrame | pl.DataFrame, \
        model_path:str=None):
        pass
    

    def _train_and_predict(self, anchor, model_path: str = None):
        X_train, y_train, X_test, y_test = self._split_data(anchor)
        if len(X_train) < self.window_size:
            self.logger.warning(
                "At anchor {}, training rows {} < window_size {} (insufficient history "
                "or many NaN future_date tail rows).",
                anchor,
                len(X_train),
                self.window_size,
            )

        if self.walkforward_debug and self.mode == "pd" and len(y_train) and len(X_test):
            tr_d0 = y_train[self.date_col].min()
            tr_d1 = y_train[self.date_col].max()
            te_d0 = X_test[self.date_col].iloc[0]
            te_d1 = X_test[self.date_col].iloc[-1]
            if self.future_period_used == 0:
                no_cal_leak = tr_d1 < te_d0
                hint = (
                    "OK: train uses only dates strictly before first test date."
                    if no_cal_leak
                    else "WARN: same-day or overlapping calendar between train tail and test head."
                )
            else:
                no_cal_leak = None
                hint = (
                    "k>0: train may include rows on/after anchor when future_date<=anchor "
                    "(label already realized); check future_date vs anchor, not only calendar date."
                )
            line = (
                f"[walkforward] anchor={anchor} | y_train calendar [{tr_d0} .. {tr_d1}] n={len(y_train)} "
                f"| X_test calendar [{te_d0} .. {te_d1}] n={len(X_test)} | {hint}"
            )
            self.logger.info(line)
            print(line)

        self._train_function(X_train, y_train, model_path)
        y_pred_list = []
        y_test_list = []
        date_list = []
        for i in range(self.rolling_freq):
            x_row = X_test.iloc[i]
            y_val = y_test.iloc[i].get(self.target_col)
            pred = self._predict_function(x_row, model_path)
            y_pred_list.append(pred)
            y_test_list.append(y_val)
            date_list.append(x_row.get(self.date_col))
            if self.walkforward_debug:
                row_line = (
                    f"  -> step i={i} pred_date={x_row.get(self.date_col)} "
                    f"y_test={y_val!r} y_pred={pred!r}"
                )
                self.logger.info(row_line)
                print(row_line)
        return y_pred_list, y_test_list, date_list

    def _rolling_train_predict(
        self,
        start: str = None,
        end: str = None,
        use_tqdm: bool = True,
    ):
        anchors = self.target[self.date_col]
        if start is not None:
            anchors = anchors[anchors >= pd.to_datetime(start)]
        if end is not None:
            anchors = anchors[anchors <= pd.to_datetime(end)]
        anchors = anchors.sort_values()
        if len(anchors) == 0:
            self.logger.warning(
                "No anchor dates in [{}, {}]; returning empty result.", start, end
            )
            return pd.DataFrame(columns=["date", "y_pred", "y_test", "diff"])

        self.logger.info(
            "Rolling over {} anchor dates (start={}, end={}).",
            len(anchors),
            start,
            end,
        )
        if self.walkforward_debug:
            preview = [str(x) for x in anchors.head(8).tolist()]
            more = "" if len(anchors) <= 8 else f" ... (+{len(anchors) - 8} more)"
            dbg = f"[walkforward] anchor list (first 8){more}: {preview}"
            self.logger.info(dbg)
            print(dbg)

        y_preds = []
        y_tests = []
        dates = []
        loop = anchors.tolist()
        if use_tqdm:
            loop = _tqdm(
                loop,
                desc=f"{self.name} rolling",
                unit="anchor",
                total=len(loop),
            )
        for anchor in loop:
            y_pred, y_test, pred_dates = self._train_and_predict(anchor)
            y_preds.append(y_pred)
            y_tests.append(y_test)
            dates.append(pred_dates)

        flat_dates = [item for sublist in dates for item in sublist]
        flat_pred = [item for sublist in y_preds for item in sublist]
        flat_test = [item for sublist in y_tests for item in sublist]
        df = pd.DataFrame({"date": flat_dates, "y_pred": flat_pred, "y_test": flat_test})
        df["diff"] = df["y_pred"] - df["y_test"]
        return df
    
    def _evaluate(self,
                  res: pd.DataFrame,
                  metrics: list = None,
                  metric_params: dict = None):
        """
        Runs evaluation by composing _summary_of_results, _show_diff_distribution,
        and _show_timeseries according to ``metrics``.

        Args:
            res (pd.DataFrame): Should include columns 'y_pred', 'y_test', 'diff', and 'date'.
            metrics (list): One or more of:
                - "summary": numeric summary via _summary_of_results
                - "diff_distribution": histogram via _show_diff_distribution
                - "timeseries": line plot via _show_timeseries
            metric_params (dict): Optional nested dicts, e.g.
                ``{"diff_distribution": {"path": ..., "bins": 100},
                   "timeseries": {
                       "path": ..., "diff_allowed": False,
                       "forward_period_used": optional override for plot captions (else self.future_period_used),
                       "plot_train_history": True,
                       "train_history": DataFrame(columns date, y),
                       "train_history_from": optional str/Timestamp — only plot train_history with date >= this,
                   }}``

        Returns:
            dict: For "summary", the summary dict; for plot metrics, the save path used.
        """
        if metrics is None:
            metrics = ['summary']
        if metric_params is None:
            metric_params = {}

        known = {'summary', 'diff_distribution', 'timeseries'}
        results = {}

        for metric in metrics:
            if metric not in known:
                raise ValueError(
                    f"Unknown metric: {metric}. Expected one of {sorted(known)}."
                )
            if metric == 'summary':
                results['summary'] = self._summary_of_results(res)
            elif metric == 'diff_distribution':
                p = metric_params.get('diff_distribution', {})
                path = p.get('path')
                bins = p.get('bins', 100)
                out = path if path is not None else f'./img/{self.name}_diff_distribution.png'
                self._show_diff_distribution(res, path=path, bins=bins)
                results['diff_distribution'] = out
            else:
                p = metric_params.get('timeseries', {})
                path = p.get('path')
                diff_allowed = p.get('diff_allowed', False)
                train_history = p.get('train_history')
                plot_train = p.get('plot_train_history', train_history is not None)
                train_history_from = p.get('train_history_from')
                out = path if path is not None else f'./img/{self.name}_timeseries.png'
                self._show_timeseries(
                    res,
                    path=path,
                    diff_allowed=diff_allowed,
                    train_history=train_history if plot_train else None,
                    train_history_from=train_history_from,
                    forward_period_used=p.get(
                        "forward_period_used", self.future_period_used
                    ),
                )
                results['timeseries'] = out

        return results


    def _summary_of_results(self, res:pd.DataFrame):
        """
        Summary of the results: mse, rmse, max of diff, 75% quantile, 50% quanntile, 25% quantile ect.
        """
        mse = ((res['diff']) ** 2).mean()
        rmse = ((res['diff']) ** 2).mean() ** 0.5
        diff = (res['diff']).abs()
        max_diff = diff.max()
        quantiles = diff.quantile([0.75, 0.5, 0.25])
        summary = {
            'mse': mse,
            'rmse': rmse,
            'max_diff': max_diff,
            '75%_quantile': quantiles.loc[0.75],
            '50%_quantile': quantiles.loc[0.5],
            '25%_quantile': quantiles.loc[0.25],
        }
        return summary

    def _show_diff_distribution(self, res: pd.DataFrame, path: str = None, bins: int = 100):
        if path is None:
            path = f'./img/{self.name}_diff_distribution.png'
        _ensure_parent_dir(path)
        self.logger.info("Saving diff distribution plot to {}", path)
        for style in ("seaborn-v0_8-whitegrid", "ggplot", "default"):
            try:
                if style == "default":
                    plt.style.use("default")
                else:
                    plt.style.use(style)
                break
            except OSError:
                continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(res["diff"], bins=bins, color="#3498db", alpha=0.75, density=True, edgecolor="white")
        ax.set_title(f"Prediction error (pred − actual) — {self.name}", fontsize=12)
        ax.set_xlabel("Error")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _show_timeseries(
        self,
        res_df: pd.DataFrame,
        path: str = None,
        diff_allowed: bool = False,
        train_history: pd.DataFrame | None = None,
        train_history_from: str | pd.Timestamp | None = None,
        forward_period_used: int | None = None,
    ):
        """
        Pred vs actual on ``res_df``; optional ``train_history`` (columns date, y)
        draws the realized training targets before the first forecast date.
        If ``train_history_from`` is set, only points with ``date >= train_history_from`` are drawn.
        If ``diff_allowed``, diff uses a twin y-axis.
        ``forward_period_used`` (k) is only for axis titles / legend text.
        """
        if path is None:
            path = f'./img/{self.name}_timeseries.png'
        _ensure_parent_dir(path)
        self.logger.info("Saving time series plot to {}", path)

        for style in ("seaborn-v0_8-whitegrid", "ggplot", "default"):
            try:
                if style == "default":
                    plt.style.use("default")
                else:
                    plt.style.use(style)
                break
            except OSError:
                continue

        c_train, c_act, c_pred, c_diff = "#95a5a6", "#e74c3c", "#2980b9", "#7f8c8d"
        fig, ax1 = plt.subplots(figsize=(12, 5.5))

        k = forward_period_used
        if k is None:
            y_lbl = "Target"
            act_lbl = "actual y"
            title_suffix = "as-of T vs forward-window y"
        elif k == 0:
            y_lbl = "Target"
            act_lbl = "actual y (same bar)"
            title_suffix = "as-of T, k=0 (same-bar label)"
        else:
            y_lbl = "Target"
            act_lbl = f"actual y (uses next k={k} rows after T)"
            title_suffix = f"as-of T, label spans next k={k} trading steps"

        if train_history is not None and len(train_history):
            th = train_history.sort_values("date").copy()
            th["date"] = pd.to_datetime(th["date"])
            if train_history_from is not None:
                t0 = pd.to_datetime(train_history_from)
                th = th.loc[th["date"] >= t0]
            train_lbl = "train target (realized, ≤ anchor)"
            if train_history_from is not None and len(th):
                train_lbl = f"train target (from {pd.Timestamp(train_history_from).date()})"
            if len(th):
                ax1.plot(
                    th["date"],
                    th["y"],
                    color=c_train,
                    linewidth=1.4,
                    linestyle="--",
                    label=train_lbl,
                    alpha=0.9,
                )
        ax1.plot(
            res_df["date"],
            res_df["y_test"],
            color=c_act,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=act_lbl,
        )
        ax1.plot(
            res_df["date"],
            res_df["y_pred"],
            color=c_pred,
            linewidth=1.8,
            marker="s",
            markersize=3,
            label="predicted",
            alpha=0.9,
        )
        if len(res_df):
            x0 = pd.Timestamp(res_df["date"].iloc[0])
            ax1.axvline(x0, color="#bdc3c7", linestyle=":", linewidth=1.2, label="first anchor")

        ax1.set_title(f"Walk-forward: {self.name} — {title_suffix}", fontsize=12)
        ax1.set_xlabel("As-of date (decision day T, after close)")
        ax1.set_ylabel(y_lbl)
        ax1.legend(loc="upper left", framealpha=0.92, fontsize=9)
        ax1.grid(True, alpha=0.35)

        if diff_allowed:
            ax2 = ax1.twinx()
            ax2.fill_between(
                res_df["date"],
                res_df["diff"],
                0,
                color=c_diff,
                alpha=0.15,
            )
            ax2.plot(
                res_df["date"],
                res_df["diff"],
                color=c_diff,
                linewidth=1.0,
                linestyle="-",
                label="error (pred−act)",
                alpha=0.85,
            )
            ax2.set_ylabel("Error", color=c_diff)
            ax2.tick_params(axis="y", labelcolor=c_diff)
            ax2.legend(loc="upper right", framealpha=0.92, fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        metrics: list | None = None,
        metric_params: dict | None = None,
        use_tqdm: bool = True,
    ):
        if metrics is None:
            metrics = ["summary", "diff_distribution", "timeseries"]
        self.logger.info("Data preprocessing...")
        self._data_preprocessing()

        self.logger.info("Rolling train and predict...")
        res = self._rolling_train_predict(start, end, use_tqdm=use_tqdm)
        self.logger.info("Evaluation rows: {}", len(res))
        self.logger.info("Evaluating...")
        results = self._evaluate(res, metrics, metric_params)
        self.logger.info("Results: {}", results)
        return results


    