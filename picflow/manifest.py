"""处理清单模块 - 记录输入/输出/尺寸/格式/耗时/抽检/失败，支持增量跳过与CSV导出"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ManifestEntry:
    input_path: str
    output_paths: List[str] = field(default_factory=list)
    sizes: List[Dict[str, Any]] = field(default_factory=list)
    format: str = "webp"
    quality: int = 75
    duration_ms: int = 0
    is_sample: bool = False
    sample_position: str = ""
    success: bool = True
    error_category: str = ""
    error_message: str = ""
    sku_key: str = ""
    sku_dir: str = ""
    mtime: float = 0.0
    file_size: int = 0


@dataclass
class ManifestMeta:
    created_at: str = ""
    preset: str = ""
    input_dir: str = ""
    output_dir: str = ""
    total_input: int = 0
    total_output: int = 0
    total_success: int = 0
    total_failed: int = 0
    total_samples: int = 0
    total_duration_ms: int = 0


class Manifest:
    """处理清单 - 读写 manifest.json，增量跳过，CSV导出"""

    FILENAME = "manifest.json"

    def __init__(self) -> None:
        self.meta: ManifestMeta = ManifestMeta()
        self.entries: List[ManifestEntry] = []

    @staticmethod
    def _path_str(p: Path) -> str:
        return str(p).replace("\\", "/")

    def add_entry(
        self,
        input_path: Path,
        output_paths: List[Path],
        sizes: List[Dict[str, Any]],
        fmt: str,
        quality: int,
        duration_ms: int,
        is_sample: bool,
        sample_position: str,
        success: bool,
        error_category: str,
        error_message: str,
        sku_key: str,
        sku_dir: str,
    ) -> None:
        try:
            stat = input_path.stat()
            mtime = stat.st_mtime
            file_size = stat.st_size
        except OSError:
            mtime = 0.0
            file_size = 0

        self.entries.append(ManifestEntry(
            input_path=self._path_str(input_path),
            output_paths=[self._path_str(p) for p in output_paths],
            sizes=sizes,
            format=fmt,
            quality=quality,
            duration_ms=duration_ms,
            is_sample=is_sample,
            sample_position=sample_position,
            success=success,
            error_category=error_category,
            error_message=error_message,
            sku_key=sku_key,
            sku_dir=sku_dir,
            mtime=mtime,
            file_size=file_size,
        ))

    def finalize(
        self,
        preset: str,
        input_dir: Path,
        output_dir: Path,
        total_duration_ms: int,
    ) -> None:
        self.meta.created_at = datetime.now().isoformat()
        self.meta.preset = preset
        self.meta.input_dir = self._path_str(input_dir)
        self.meta.output_dir = self._path_str(output_dir)
        self.meta.total_input = len(self.entries)
        self.meta.total_output = sum(len(e.output_paths) for e in self.entries)
        self.meta.total_success = sum(1 for e in self.entries if e.success)
        self.meta.total_failed = sum(1 for e in self.entries if not e.success)
        self.meta.total_samples = sum(1 for e in self.entries if e.is_sample)
        self.meta.total_duration_ms = total_duration_ms

    def save(self, output_dir: Path) -> Path:
        path = output_dir / self.FILENAME
        data = {
            "meta": asdict(self.meta),
            "entries": [asdict(e) for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, output_dir: Path) -> Optional[Manifest]:
        path = output_dir / cls.FILENAME
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = cls()
            meta_data = data.get("meta", {})
            m.meta = ManifestMeta(**{k: meta_data.get(k, "") for k in ManifestMeta.__dataclass_fields__})
            for entry_data in data.get("entries", []):
                filtered = {k: entry_data.get(k, "") for k in ManifestEntry.__dataclass_fields__}
                if isinstance(filtered.get("output_paths"), str):
                    filtered["output_paths"] = [filtered["output_paths"]]
                if isinstance(filtered.get("sizes"), str):
                    filtered["sizes"] = []
                m.entries.append(ManifestEntry(**filtered))
            return m
        except Exception:
            return None

    def get_processed_set(self) -> Dict[str, tuple]:
        result: Dict[str, tuple] = {}
        for e in self.entries:
            if e.success and not e.is_sample:
                result[e.input_path] = (e.mtime, e.file_size)
        return result

    def should_skip(self, input_path: Path) -> bool:
        key = self._path_str(input_path)
        if key not in self._processed_cache:
            return False
        cached_mtime, cached_size = self._processed_cache[key]
        try:
            stat = input_path.stat()
            return abs(stat.st_mtime - cached_mtime) < 1.0 and stat.st_size == cached_size
        except OSError:
            return False

    def build_skip_cache(self) -> None:
        self._processed_cache = self.get_processed_set()

    def export_csv(self, csv_path: Path) -> Path:
        fieldnames = [
            "input_path", "output_paths", "sku_key", "sku_dir",
            "sizes", "format", "quality", "duration_ms",
            "is_sample", "sample_position",
            "success", "error_category", "error_message",
            "mtime", "file_size",
        ]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                row = asdict(e)
                row["output_paths"] = "; ".join(row["output_paths"])
                row["sizes"] = "; ".join(
                    f'{s.get("name","")}{s.get("width","")}x{s.get("height","")}{s.get("suffix","")}'
                    for s in row["sizes"]
                )
                writer.writerow(row)
        return csv_path

    def get_summary(self) -> Dict[str, Any]:
        sku_counts: Dict[str, int] = {}
        size_counts: Dict[str, int] = {}
        sample_positions: List[str] = []
        error_counts: Dict[str, int] = {}

        for e in self.entries:
            if e.sku_key:
                sku_counts[e.sku_key] = sku_counts.get(e.sku_key, 0) + 1
            for s in e.sizes:
                key = f'{s.get("width","")}x{s.get("height","")}'
                size_counts[key] = size_counts.get(key, 0) + 1
            if e.is_sample:
                sample_positions.append(f"{e.input_path} @ {e.sample_position}")
            if not e.success and e.error_category:
                error_counts[e.error_category] = error_counts.get(e.error_category, 0) + 1

        return {
            "total_input": self.meta.total_input,
            "total_output_files": self.meta.total_output,
            "total_success": self.meta.total_success,
            "total_failed": self.meta.total_failed,
            "total_samples": self.meta.total_samples,
            "total_duration_ms": self.meta.total_duration_ms,
            "sku_counts": sku_counts,
            "size_counts": size_counts,
            "sample_positions": sample_positions,
            "error_counts": error_counts,
        }
