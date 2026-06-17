"""图片去重模块 - 感知哈希计算与重复检测"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import exifread
import imagehash
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


@dataclass
class ImageHashInfo:
    path: Path
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    ahash: imagehash.ImageHash
    resolution: Tuple[int, int]
    exif_tags_count: int
    file_size: int

    @property
    def score(self) -> float:
        w, h = self.resolution
        pixel_score = (w * h) / (5000 * 5000)
        exif_score = min(self.exif_tags_count / 50.0, 1.0)
        size_score = min(self.file_size / (10 * 1024 * 1024), 1.0)
        return pixel_score * 0.5 + exif_score * 0.3 + size_score * 0.2


@dataclass
class DuplicatePair:
    keep: ImageHashInfo
    remove: ImageHashInfo
    distance: int
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, 1.0 - (self.distance / 64.0))


@dataclass
class DuplicateGroup:
    images: List[ImageHashInfo]
    keep_index: int
    distances: List[int]


class ImageHasher:
    """图片哈希计算器与重复检测器"""

    def __init__(self, threshold: int = 8, hash_size: int = 8) -> None:
        self.threshold = threshold
        self.hash_size = hash_size

    def compute_hashes(self, image_path: Path) -> Optional[ImageHashInfo]:
        try:
            with Image.open(image_path) as img:
                resolution = img.size
                phash = imagehash.phash(img, hash_size=self.hash_size)
                dhash = imagehash.dhash(img, hash_size=self.hash_size)
                ahash = imagehash.average_hash(img, hash_size=self.hash_size)

            exif_count = 0
            try:
                with open(image_path, "rb") as f:
                    tags = exifread.process_file(f, details=False)
                    exif_count = len(tags) if tags else 0
            except Exception:
                exif_count = 0

            return ImageHashInfo(
                path=image_path,
                phash=phash,
                dhash=dhash,
                ahash=ahash,
                resolution=resolution,
                exif_tags_count=exif_count,
                file_size=image_path.stat().st_size,
            )
        except Exception:
            return None

    def hamming_distance(
        self, a: ImageHashInfo, b: ImageHashInfo
    ) -> Tuple[int, int, int]:
        return (a.phash - b.phash, a.dhash - b.dhash, a.ahash - b.ahash)

    def is_duplicate(self, a: ImageHashInfo, b: ImageHashInfo) -> Tuple[bool, int]:
        p_dist, d_dist, a_dist = self.hamming_distance(a, b)
        avg_dist = (p_dist * 0.5 + d_dist * 0.3 + a_dist * 0.2)
        is_dup = p_dist < self.threshold or avg_dist < self.threshold * 0.9
        return is_dup, int(avg_dist)

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    ) -> List[ImageHashInfo]:
        image_paths: List[Path] = []
        if recursive:
            for root, _, files in os.walk(directory):
                for fname in files:
                    p = Path(root) / fname
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        image_paths.append(p)
        else:
            for fname in os.listdir(directory):
                p = directory / fname
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(p)

        hashes: List[ImageHashInfo] = []
        total = len(image_paths)
        for idx, path in enumerate(image_paths, 1):
            info = self.compute_hashes(path)
            if info is not None:
                hashes.append(info)
            if progress_callback:
                progress_callback(idx, total, path)
        return hashes

    def find_duplicates(
        self,
        hash_infos: List[ImageHashInfo],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DuplicatePair]:
        pairs: List[DuplicatePair] = []
        total = len(hash_infos)

        for i in range(total):
            if progress_callback:
                progress_callback(i + 1, total)
            for j in range(i + 1, total):
                a, b = hash_infos[i], hash_infos[j]
                is_dup, dist = self.is_duplicate(a, b)
                if is_dup:
                    keep = a if a.score >= b.score else b
                    remove = b if keep is a else a
                    pairs.append(DuplicatePair(keep=keep, remove=remove, distance=dist))

        return pairs

    def group_duplicates(
        self, pairs: List[DuplicatePair]
    ) -> List[DuplicateGroup]:
        parent: Dict[Path, Path] = {}

        def find(x: Path) -> Path:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: Path, b: Path) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        path_to_info: Dict[Path, ImageHashInfo] = {}
        for p in pairs:
            path_to_info[p.keep.path] = p.keep
            path_to_info[p.remove.path] = p.remove
            union(p.keep.path, p.remove.path)

        groups: Dict[Path, List[ImageHashInfo]] = {}
        for path, info in path_to_info.items():
            root = find(path)
            if root not in groups:
                groups[root] = []
            groups[root].append(info)

        result: List[DuplicateGroup] = []
        for _, imgs in groups.items():
            if len(imgs) < 2:
                continue
            best_idx = 0
            best_score = -1.0
            for idx, img in enumerate(imgs):
                if img.score > best_score:
                    best_score = img.score
                    best_idx = idx

            dists: List[int] = []
            for idx, img in enumerate(imgs):
                if idx != best_idx:
                    is_dup, dist = self.is_duplicate(imgs[best_idx], img)
                    dists.append(dist)

            result.append(
                DuplicateGroup(
                    images=imgs, keep_index=best_idx, distances=dists
                )
            )
        return result

    def organize_duplicates(
        self,
        groups: List[DuplicateGroup],
        output_dir: Path,
        use_symlinks: bool = True,
    ) -> Dict[str, int]:
        stats = {"groups": len(groups), "linked": 0, "moved": 0}
        dupes_dir = output_dir / "dupes"

        for g_idx, group in enumerate(groups, 1):
            group_dir = dupes_dir / f"group_{g_idx:04d}"
            keep_dir = group_dir / "keep"
            remove_dir = group_dir / "remove"

            keep_dir.mkdir(parents=True, exist_ok=True)
            remove_dir.mkdir(parents=True, exist_ok=True)

            for idx, img in enumerate(group.images):
                dest_dir = keep_dir if idx == group.keep_index else remove_dir
                dest = dest_dir / img.path.name

                counter = 1
                original_dest = dest
                while dest.exists():
                    stem = original_dest.stem
                    suffix = original_dest.suffix
                    dest = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                try:
                    if use_symlinks:
                        if dest.exists():
                            dest.unlink()
                        os.symlink(img.path, dest)
                        stats["linked"] += 1
                    else:
                        shutil.copy2(img.path, dest)
                        stats["moved"] += 1
                except OSError:
                    shutil.copy2(img.path, dest)
                    stats["moved"] += 1

        return stats

    def get_skip_set(self, pairs: List[DuplicatePair]) -> set:
        return {p.remove.path for p in pairs}
