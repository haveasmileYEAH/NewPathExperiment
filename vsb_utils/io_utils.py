# common/io_utils.py
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd


def ensure_dir(path: str | Path) -> None:
    """
    如果 path 是文件路径，则创建其父目录；
    如果 path 是目录路径，则直接创建目录。
    """
    p = Path(path)
    if p.suffix:  # 有后缀，视为文件路径
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    ensure_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    ensure_dir(path)
    df.to_csv(path, index=index)
