from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_DATA_CANDIDATES, PROCESSED_DATA_DIR, RANDOM_STATE
from src.utils import ensure_dir


def resolve_data_path(explicit_path: Optional[str] = None) -> Path:
    candidates: Iterable[Path]
    candidates = [Path(explicit_path)] if explicit_path else DEFAULT_DATA_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No supported dataset file found.")


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    path = resolve_data_path(data_path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]

    if "url" not in frame.columns:
        raise ValueError("Dataset must include a 'url' column.")

    label_column = _resolve_label_column(frame)
    frame = frame[["url", label_column]].rename(columns={label_column: "label"})
    frame["url"] = frame["url"].astype(str).str.strip()
    frame = frame[frame["url"] != ""]
    frame["label"] = frame["label"].apply(_normalize_label)
    frame = frame.dropna(subset=["label"]).reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    return frame


def train_val_test_split(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame["label"],
        random_state=RANDOM_STATE,
    )
    adjusted_val_size = val_size / (1 - test_size)
    train_frame, val_frame = train_test_split(
        train_frame,
        test_size=adjusted_val_size,
        stratify=train_frame["label"],
        random_state=RANDOM_STATE,
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def save_splits(train_frame: pd.DataFrame, val_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    ensure_dir(PROCESSED_DATA_DIR)
    train_frame.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_frame.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_frame.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def _resolve_label_column(frame: pd.DataFrame) -> str:
    for column in ("label", "status", "target", "class"):
        if column in frame.columns:
            return column
    raise ValueError("Dataset must include one of: label, status, target, class.")


def _normalize_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"phishing", "malicious", "bad", "1", "true"}:
            return 1
        if lowered in {"legitimate", "benign", "good", "0", "false"}:
            return 0
        return None
    if value in (0, 1):
        return int(value)
    return None
