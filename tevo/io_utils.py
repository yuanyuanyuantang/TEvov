"""小工具：原子写文件。

多 worker（多个 Ray actor 并发调 TEvo 服务）场景下，记忆库/画像库会被并发写；
`Path.write_text` 是"截断 + 写入"，并发读方可能读到半截 JSON。这里统一改成
"写临时文件 + os.replace"，保证读方永远看到完整内容。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    """原子写：先写同目录临时文件再 os.replace 覆盖目标。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
