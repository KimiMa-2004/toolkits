import polars as pl
import logging
from typing import Union, Tuple, List, Optional, Any
from toolkits.logger import get_logger
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import DataLoader, Dataset


class TimeSeriesSplitter:
    """
    基于 Polars 的时间序列训练/测试分割器，通过 `target_latest_shift` 处理数据泄露问题。

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        必须包含 `date_col` 和 `target_col`。特征/资产列是可选的。
    date_col : str
        `df` 中日期列的名称。
    target_col : str
        `df` 中目标列的名称。
    target_latest_shift : int
        目标值向未来看的交易日数。
        例如，预测 T+1 到 T+5 的回报 → target_latest_shift = 5。
        这构建了 `latest_date = calendar[date].shift(-target_latest_shift)`，
        因此日期为 T 的行有 latest_date=T+5，意味着其目标值依赖于直到 T+5 的数据。
    calendar : str | pl.DataFrame | pl.LazyFrame
        路径(parquet/csv)或至少包含一个日期列的DataFrame(交易日历)。
        用于通过交易日感知的偏移计算`latest_date`。
    feature_cols : List[str] | None
        要使用的特征列。如果为None，则使用除日期/目标/资产外的所有列。
    asset_col : str | None
        如果为None，将df视为单资产。否则按此列分组。
    train_window : int
        有效训练样本的数量(seq_len对齐后的行数)。
    seq_len : int
        序列模型(LSTM, Transformer等)的序列长度。
        对于树/MLP模型设为1。
    logger : logging.Logger | None
        可选的日志记录器。默认为名为'TimeSeriesSplitter'的新日志记录器。
    """

    def __init__(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        date_col: str,
        target_col: str,
        target_latest_shift: int,
        calendar: Union[str, pl.DataFrame, pl.LazyFrame],
        feature_cols: Optional[List[str]] = None,
        asset_col: Optional[str] = None,
        train_window: int = 752,
        seq_len: int = 252,
        logger: Optional[logging.Logger] = None,
        ):
        self.date_col = date_col
        self.target_col = target_col
        self.target_latest_shift = target_latest_shift
        self.feature_cols = feature_cols
        self.asset_col = asset_col
        self.train_window = train_window
        self.seq_len = seq_len
        self.logger = logger or get_logger("TimeSeriesSplitter")

        # 将LazyFrame转换为DataFrame用于初始化 (我们保留初始数据为eager模式以便高效查询)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # 检查输入DataFrame是否为空
        if df.is_empty():
            self.logger.warning("输入DataFrame为空。分割将返回空DataFrame。")

        # 如果未指定特征列，则确定特征列
        if self.feature_cols is None:
            exclude = {date_col, target_col}
            if asset_col:
                exclude.add(asset_col)
            self.feature_cols = [c for c in df.columns if c not in exclude]

        # 准备日历以进行高效的日期查找
        self.calendar = self._load_and_prepare_calendar(calendar)
        
        # 检查日历是否为空
        if len(self.calendar) == 0:
            self.logger.warning("日历为空。日期查找将失败。")

        # 准备DataFrame - 确保日期列为datetime类型并按日期排序
        # 移除了.reset_index(drop=True)，因为这是Pandas的方法，Polars DataFrame没有这个方法
        self.df = (df.with_columns(pl.col(self.date_col).cast(pl.Date))
                .sort(self.date_col))

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_calendar(self, calendar: Union[str, pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
        """
        从文件或DataFrame加载日历。
        """
        if isinstance(calendar, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(calendar, pl.LazyFrame):
                return calendar.collect()
            return calendar.clone()
        
        try:
            return pl.read_parquet(calendar)
        except Exception:
            pass
        
        try:
            return pl.read_csv(calendar)
        except Exception as e:
            self.logger.error(f"加载日历失败: {e}")
            raise ValueError(f"无法读取日历文件: {calendar}") from e

    def _load_and_prepare_calendar(self, calendar: Union[str, pl.DataFrame, pl.LazyFrame]) -> pl.Series:
        """
        准备用于高效日期查找的日历。
        
        返回一个pl.Series，其中索引是位置，值是交易日期。
        这允许通过特定交易日数偏移高效查找日期（O(log n)）。
        """
        cal = self._load_calendar(calendar)
        if cal.is_empty():
            self.logger.warning("日历为空。日期查找将失败。")
            return pl.Series("date", [], dtype=pl.Date)
            
        cal_date_col = cal.columns[0]
        dates = (cal.select(pl.col(cal_date_col).cast(pl.Date).alias("date"))
                 .unique()
                 .sort("date")
                 .get_column("date"))
        
        return dates  # pl.Series of Dates

    def _lookup_start(self, date: Union[str, pl.Date], offset: int) -> Optional[pl.Date]:
        """
        查找在'date'之前'offset'个交易日的日期。
        """
        from datetime import datetime
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        if len(self.calendar) == 0:
            self.logger.warning("日历为空，无法查找日期。")
            return None

        # 使用 search_sorted 找到插入位置
        pos = self.calendar.search_sorted(date)

        if pos == len(self.calendar):
            pos = len(self.calendar) - 1
        elif pos > 0 and self.calendar[pos] != date:
            pos -= 1

        start_pos = pos - offset
        if start_pos < 0:
            self.logger.warning(
                f"{date}之前的历史数据不足: 需要{offset}个交易日, "
                f"只有{pos+1}个可用(最早: {self.calendar[0]})。"
            )
            return None

        return self.calendar[start_pos]

    def _lookup_forward(self, date: Union[str, pl.Date], offset: int) -> Optional[pl.Date]:
        """查找在 ``date`` 之后 ``offset`` 个交易日的日期。"""
        from datetime import datetime
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if len(self.calendar) == 0:
            self.logger.warning("日历为空，无法向前查找日期。")
            return None
        pos = self.calendar.search_sorted(date)
        if pos == len(self.calendar):
            return None
        if self.calendar[pos] != date:
            # date 不是交易日时，search_sorted 给的是下一个交易日位置
            base_pos = pos
        else:
            base_pos = pos
        target_pos = base_pos + offset
        if target_pos >= len(self.calendar):
            return None
        return self.calendar[target_pos]

    def _convert_to_date(self, date_val):
        """将各种日期格式转换为pl.Date对象"""
        if isinstance(date_val, str):
            # 使用datetime标准库转换字符串到日期
            from datetime import datetime
            try:
                # 尝试ISO格式 YYYY-MM-DD
                return datetime.strptime(date_val, "%Y-%m-%d").date()
            except ValueError:
                try:
                    # 尝试其他常见格式
                    return datetime.strptime(date_val, "%Y/%m/%d").date()
                except ValueError:
                    # 如果都失败了，使用pandas更灵活的解析器
                    import pandas as pd
                    return pd.to_datetime(date_val).date()
        return date_val  # 如果已经是日期对象，直接返回


    def check_data_sufficiency(self, date: Union[str, pl.Date]) -> bool:
        """
        检查是否有足够的数据在给定日期执行拆分。
        """
        date = self._convert_to_date(date)
        
        # 检查日历是否有足够的日期
        if len(self.calendar) < (self.train_window + self.seq_len):
            self.logger.warning(
                f"日历仅有{len(self.calendar)}个日期, "
                f"至少需要{self.train_window + self.seq_len}个。"
            )
            return False
        
        # 检查日期是否在日历范围内
        if date < self.calendar[0]:
            self.logger.warning(f"请求的日期{date}在日历开始日期{self.calendar[0]}之前")
            return False
        
        # 检查我们是否有足够的历史
        test_start = self._lookup_start(date, self.seq_len - 1)
        train_raw_start = self._lookup_start(date, self.train_window + self.seq_len - 2)
        
        if test_start is None or train_raw_start is None:
            return False
        
        # 检查我们是否在所需的日期范围内有数据
        date_counts = (self.df
                    .filter((pl.col(self.date_col) >= train_raw_start) & 
                            (pl.col(self.date_col) <= date))
                    .select(pl.count())
                    .item())
        
        if date_counts == 0:
            self.logger.warning(f"在{date}所需的日期范围内没有可用数据")
            return False
        
        return True

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------

    def split(self, date: Union[str, pl.Date], test_with_target: bool = True) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        from datetime import datetime
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        # 计算训练日期范围
        train_raw_start = self._lookup_start(date, self.target_latest_shift+self.train_window + self.seq_len - 2)
        train_end = self._lookup_start(date, self.target_latest_shift)

        # 计算测试日期范围
        test_start = self._lookup_start(date, self.seq_len - 1)
        test_end = date
        if test_with_target:
            next_day = self._lookup_forward(date, 1)
            if next_day is not None:
                test_end = next_day

        # 处理数据不足情况
        if train_raw_start is None:
            earliest_date = self.calendar[0] if len(self.calendar) > 0 else None
            self.logger.warning(
                f"请求的训练数据在日历开始之前。使用最早可用日期: {earliest_date}"
            )
            train_raw_start = earliest_date

        if train_end is None or test_start is None:
            self.logger.error(f"数据不足，无法在日期={date}拆分。至少需要{self.seq_len}个交易日。")
            raise ValueError(f"数据不足，无法在日期={date}拆分")

        # 构建 LazyFrame
        train = (self.df.lazy()
                .filter((pl.col(self.date_col) >= train_raw_start) & 
                        (pl.col(self.date_col) <= train_end)))

        test = (self.df.lazy()
            .filter((pl.col(self.date_col) >= test_start) & 
                    (pl.col(self.date_col) <= test_end)))

        return train, test, self.seq_len

    # 添加一个辅助方法，当用户需要实际执行并获取结果时使用
    def split_collect(self, date: Union[str, pl.Date]) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """与split相同，但返回实际的DataFrame而不是LazyFrame"""
        train_lazy, test_lazy = self.split(date)
        return train_lazy.collect(), test_lazy.collect(), self.seq_len



class TimeSeriesDataset(IterableDataset):
    """
    多资产时序数据集，生成 (序列, 下一日目标) 样本。

    Parameters
    ----------
    train : pl.LazyFrame
        训练集 LazyFrame，包含列: date_col, asset_col, target_col, feature_cols
    test : pl.LazyFrame
        测试集 LazyFrame (目前仅在验证时使用，本类专注于训练集生成)
    target_col : str
        目标列名
    date_col : str
        日期列名
    seq_len : int
        序列长度（历史天数）
    asset_col : str
        资产标识列名
    feature_cols : List[str] | None
        特征列名列表。若为 None，则自动使用除 date/asset/target 外的所有列。
    shuffle : bool, default=True
        是否在样本级别打乱（仅在训练时有效，测试时应设为 False）
    buffer_size : int, default=10000
        shuffle 缓冲区大小，越大随机性越强但内存占用越高。
    drop_incomplete : bool, default=True
        是否丢弃序列长度不足 seq_len+1 的资产（+1 用于目标值）
    with_target : bool, default=True
        是否在迭代时返回目标值。True -> ``(X, y)``，False -> ``(X,)``。
    target_dtype : torch.dtype, default=torch.float32
        目标张量类型。分类场景建议 ``torch.long``。
    logger : logging.Logger | None
        日志记录器
    """
    def __init__(
        self,
        train: pl.LazyFrame,
        test: pl.LazyFrame,           # 保留接口，但训练集使用 train
        target_col: str,
        date_col: str,
        seq_len: int,
        asset_col: str,
        feature_cols: Optional[List[str]] = None,
        shuffle: bool = True,
        buffer_size: int = 10000,
        drop_incomplete: bool = True,
        with_target: bool = True,
        target_dtype: torch.dtype = torch.float32,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.train_lf = train
        self.test_lf = test
        self.target_col = target_col
        self.date_col = date_col
        self.seq_len = seq_len
        self.asset_col = asset_col
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_incomplete = drop_incomplete
        self.with_target = with_target
        self.target_dtype = target_dtype
        self.logger = logger or logging.getLogger(__name__)


        # 确定特征列
        if feature_cols is None:
            # 排除日期、资产、目标列
            exclude = {date_col, asset_col, target_col}
            # 注意：LazyFrame 没有 columns 属性直接访问，需通过 collect 一个空 schema？
            # 简便方式：先获取一个样本 DataFrame 的列名
            sample_df = train.limit(1).collect()
            self.feature_cols = [c for c in sample_df.columns if c not in exclude]
        else:
            self.feature_cols = feature_cols

        # 提前收集所有资产的唯一标识，用于 worker 分片
        if self.asset_col is None:
            # 单资产模式：不按资产过滤。
            self.assets = [None]
        else:
            self.assets = (
                self.train_lf.select(pl.col(self.asset_col).unique())
                .collect()
                .get_column(self.asset_col)
                .to_list()
            )
        self.num_assets = len(self.assets)
        self._asset_arrays: dict[Any, tuple[np.ndarray, np.ndarray | None]] = {}
        self._prepare_asset_arrays()
        # Cached arrays are enough for iteration; drop Polars objects to improve pickling.
        self.train_lf = None
        self.test_lf = None
        self.logger.info(f"TimeSeriesDataset 初始化完成，资产数量: {self.num_assets}")

    def _prepare_asset_arrays(self) -> None:
        """一次性收集并缓存各资产数组，避免每个 epoch 重复 filter+collect。"""
        if self.asset_col is None:
            df_sorted = self.train_lf.sort(self.date_col).collect()
            X = df_sorted[self.feature_cols].to_numpy().astype(np.float32)
            y = df_sorted[self.target_col].to_numpy() if self.with_target else None
            self._asset_arrays[None] = (X, y)
            return

        df_sorted = self.train_lf.sort([self.asset_col, self.date_col]).collect()
        for asset_id in self.assets:
            df_asset = df_sorted.filter(pl.col(self.asset_col) == asset_id)
            X = df_asset[self.feature_cols].to_numpy().astype(np.float32)
            y = df_asset[self.target_col].to_numpy() if self.with_target else None
            self._asset_arrays[asset_id] = (X, y)

    def __getstate__(self):
        """Make dataset picklable for Windows spawn workers."""
        state = self.__dict__.copy()
        # Some logger implementations keep stream/file handles (non-picklable).
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)


    def _generate_samples_for_asset(self, asset_id) -> List[tuple]:
        """
        为单个资产生成所有 (X_seq, y) 样本。
        返回列表，每个元素为 (numpy_array_of_shape(seq_len, n_features), target_value)
        """
        X, y = self._asset_arrays.get(asset_id, (None, None))
        if X is None:
            return []

        min_required = self.seq_len + (1 if self.with_target else 0)
        if len(X) < min_required:
            if self.drop_incomplete:
                return []
            else:
                self.logger.warning(
                    f"资产 {asset_id} 数据不足 (长度 {len(X)} < {min_required})，将跳过。"
                )
                return []

        samples = []
        sample_count = len(X) - self.seq_len
        if not self.with_target:
            sample_count += 1
        for i in range(sample_count):
            seq_x = X[i : i + self.seq_len]          # shape (seq_len, F)
            if self.with_target:
                target = y[i + self.seq_len]             # 标量
                samples.append((seq_x, target))
            else:
                samples.append((seq_x,))
        return samples


    def __iter__(self):
        """
        迭代器逻辑：
        - 获取当前 worker 信息，对资产列表进行分片。
        - 按资产遍历，为每个资产生成样本。
        - 若 shuffle=True，使用缓冲区进行样本级打乱。
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # 单进程
            assets_to_process = self.assets
        else:
            # 多进程：按 worker id 分配资产子集
            per_worker = int(np.ceil(self.num_assets / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.num_assets)
            assets_to_process = self.assets[start:end]
            self.logger.debug(f"Worker {worker_id} 处理资产索引 {start}:{end}，共 {len(assets_to_process)} 个资产。")

        if self.shuffle:
            # 使用缓冲区进行样本级 shuffle
            buffer = []
            for asset in assets_to_process:
                for sample in self._generate_samples_for_asset(asset):
                    if len(buffer) >= self.buffer_size:
                        # O(1) 随机弹出：与末尾交换后 pop
                        idx = np.random.randint(0, len(buffer))
                        buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                        yield buffer.pop()
                    buffer.append(sample)
            # 清空缓冲区
            while buffer:
                idx = np.random.randint(0, len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()
        else:
            # 不打乱，顺序输出
            for asset in assets_to_process:
                for sample in self._generate_samples_for_asset(asset):
                    yield sample


    def collate_fn(self, batch: List[tuple[Any, ...]]):
        """
        自定义 collate 函数，用于 DataLoader。
        batch:
            - with_target=True: list of (X_seq, y)
            - with_target=False: list of (X_seq,)
        返回:
            - with_target=True: (X_batch, y_batch)
            - with_target=False: (X_batch,)
            X_batch: shape (batch_size, seq_len, n_features)
            y_batch: shape (batch_size,)
        """
        X_list = [item[0] for item in batch]
        X_batch = torch.from_numpy(np.stack(X_list, axis=0))  # (B, seq_len, F)
        if self.with_target:
            y_list = [item[1] for item in batch]
            y_batch = torch.tensor(y_list, dtype=self.target_dtype)  # (B,)
            return X_batch, y_batch
        return (X_batch,)


class MapTimeSeriesDataset(Dataset):
    """Map-style 时序窗口数据集，适合大规模训练与并行加载。"""

    def __init__(
        self,
        data_lf: pl.LazyFrame,
        target_col: str,
        date_col: str,
        seq_len: int,
        asset_col: Optional[str],
        feature_cols: Optional[List[str]] = None,
        with_target: bool = True,
        target_dtype: torch.dtype = torch.float32,
        logger: Optional[logging.Logger] = None,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.seq_len = seq_len
        self.asset_col = asset_col
        self.with_target = with_target
        self.target_dtype = target_dtype
        self.logger = logger or logging.getLogger(__name__)

        df_sorted = data_lf.sort([asset_col, date_col] if asset_col else [date_col]).collect()

        if feature_cols is None:
            exclude = {date_col, target_col}
            if asset_col:
                exclude.add(asset_col)
            feature_cols = [c for c in df_sorted.columns if c not in exclude]
        self.feature_cols = feature_cols

        self._asset_arrays: list[tuple[np.ndarray, np.ndarray | None]] = []
        self._sample_index: list[tuple[int, int]] = []
        self.assets: list[Any] = []

        if asset_col is None:
            parts = [df_sorted]
            self.assets = [None]
        else:
            parts = df_sorted.partition_by(asset_col, maintain_order=True)
            self.assets = [part[asset_col][0] for part in parts]

        for part in parts:
            X = part[self.feature_cols].to_numpy().astype(np.float32)
            y = part[self.target_col].to_numpy() if with_target else None
            min_required = self.seq_len + (1 if with_target else 0)
            if len(X) < min_required:
                continue

            asset_idx = len(self._asset_arrays)
            self._asset_arrays.append((X, y))
            sample_count = len(X) - self.seq_len + (0 if with_target else 1)
            for start in range(sample_count):
                self._sample_index.append((asset_idx, start))

        self.num_assets = len(self.assets)
        self.logger.info(f"MapTimeSeriesDataset 初始化完成，样本数: {len(self._sample_index)}")

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int):
        asset_idx, start = self._sample_index[idx]
        X, y = self._asset_arrays[asset_idx]
        seq_x = X[start : start + self.seq_len]
        x_tensor = torch.from_numpy(seq_x)
        if self.with_target:
            target = y[start + self.seq_len]
            y_tensor = torch.tensor(target, dtype=self.target_dtype)
            return x_tensor, y_tensor
        return (x_tensor,)

    def __getstate__(self):
        """Make dataset picklable for Windows spawn workers."""
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)






def construct_loader(
    splitter: TimeSeriesSplitter,
    date: str,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_buffer_size: int = 10000,
    train_target_dtype: torch.dtype = torch.float32,
    test_with_target: bool = True,
    test_target_dtype: Optional[torch.dtype] = None,
    persistent_workers: Optional[bool] = None,
    backend: str = "map",
) -> Tuple[DataLoader, DataLoader]:
    """
    根据分割器在给定日期拆分数据，并构造训练和测试 DataLoader。

    Parameters
    ----------
    splitter : TimeSeriesSplitter
        已初始化的分割器实例（包含数据、日历、参数等）
    date : str
        分割日期，格式如 "YYYY-MM-DD"
    batch_size : int, default=128
        DataLoader 批大小
    num_workers : int, default=0
        数据加载子进程数，0 表示主进程加载
    pin_memory : bool, default=True
        是否将张量复制到 CUDA 固定内存（GPU 训练时推荐 True）
    shuffle_buffer_size : int, default=10000
        训练集样本打乱缓冲区大小，越大随机性越强但内存占用越高
    train_target_dtype : torch.dtype, default=torch.float32
        训练目标张量类型。分类场景建议传 ``torch.long``。
    test_with_target : bool, default=True
        测试集是否返回目标。False 时仅返回 ``(X,)``，可用于无标签预测。
    test_target_dtype : torch.dtype | None, default=None
        测试目标张量类型。None 时沿用 ``train_target_dtype``。
    persistent_workers : bool | None, default=None
        当 ``num_workers > 0`` 时是否复用 worker。None 表示自动启用，可显著减少每个 epoch 起始开销。
    backend : str, default="map"
        数据集后端：``"map"``(推荐) 或 ``"iterable"``(兼容旧逻辑)。

    Returns
    -------
    train_loader : DataLoader
        训练数据加载器，每个 batch 为 (X, y)，X shape: (batch, seq_len, n_features)
    test_loader : DataLoader
        测试数据加载器，顺序加载，不打乱
    """
    # 1. 拆分数据，获得训练/测试 LazyFrame 和 seq_len
    train_lf, test_lf, seq_len = splitter.split(date, test_with_target=test_with_target)

    # 2. 获取特征列（从 splitter 中复用，避免重复计算）
    feature_cols = splitter.feature_cols

    if test_target_dtype is None:
        test_target_dtype = train_target_dtype
    if backend not in ("map", "iterable"):
        raise ValueError("backend must be 'map' or 'iterable'")

    if backend == "map":
        train_dataset = MapTimeSeriesDataset(
            data_lf=train_lf,
            target_col=splitter.target_col,
            date_col=splitter.date_col,
            seq_len=seq_len,
            asset_col=splitter.asset_col,
            feature_cols=feature_cols,
            with_target=True,
            target_dtype=train_target_dtype,
            logger=splitter.logger,
        )
        test_dataset = MapTimeSeriesDataset(
            data_lf=test_lf,
            target_col=splitter.target_col,
            date_col=splitter.date_col,
            seq_len=seq_len,
            asset_col=splitter.asset_col,
            feature_cols=feature_cols,
            with_target=test_with_target,
            target_dtype=test_target_dtype,
            logger=splitter.logger,
        )
    else:
        train_dataset = TimeSeriesDataset(
            train=train_lf,
            test=test_lf,
            target_col=splitter.target_col,
            date_col=splitter.date_col,
            seq_len=seq_len,
            asset_col=splitter.asset_col,
            feature_cols=feature_cols,
            shuffle=True,
            buffer_size=shuffle_buffer_size,
            drop_incomplete=True,
            logger=splitter.logger,
            with_target=True,
            target_dtype=train_target_dtype,
        )
        test_dataset = TimeSeriesDataset(
            train=test_lf,
            test=test_lf,
            target_col=splitter.target_col,
            date_col=splitter.date_col,
            seq_len=seq_len,
            asset_col=splitter.asset_col,
            feature_cols=feature_cols,
            shuffle=False,
            buffer_size=0,
            drop_incomplete=True,
            logger=splitter.logger,
            with_target=test_with_target,
            target_dtype=test_target_dtype,
        )

    # 5. 构造 DataLoader
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if backend == "map":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    splitter.logger.info(f"DataLoader 构建完成。backend={backend}, num_workers={num_workers}")

    return train_loader, test_loader
    