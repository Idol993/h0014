"""核心处理引擎 - EXIF纠正/智能裁剪/调色/WebP转换/重命名"""
from __future__ import annotations

import io
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import exifread
import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

logger = logging.getLogger("picflow")


class ErrorCategory(str, Enum):
    EXIF_CORRUPT = "EXIF损坏"
    OUT_OF_MEMORY = "内存不足"
    UNSUPPORTED_FORMAT = "格式不支持"
    WRITE_PERMISSION = "输出路径无写权限"
    UNKNOWN = "未知错误"


@dataclass
class ProcessError:
    path: Path
    category: ErrorCategory
    message: str
    stacktrace: str


@dataclass
class ProcessTask:
    input_path: Path
    rel_path: Path
    output_dir: Path
    index: int
    sku_fields: Dict[str, str] = field(default_factory=dict)
    extra_sizes: Optional[List[Dict[str, Any]]] = None
    sku_group_dir: Optional[Path] = None
    is_sample: bool = False
    sample_position: str = ""
    task_id: int = -1


@dataclass
class ProcessResult:
    input_path: Path
    task_index: int = -1
    success: bool = False
    output_paths: List[Path] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[ProcessError] = None
    skipped: bool = False
    skip_reason: str = ""
    dry_run_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BatchStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: int = 0
    errors: List[ProcessError] = field(default_factory=list)
    error_counts: Dict[ErrorCategory, int] = field(default_factory=dict)

    def add_error(self, err: ProcessError) -> None:
        self.errors.append(err)
        self.error_counts[err.category] = self.error_counts.get(err.category, 0) + 1


SUPPORTED_INPUT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _classify_error(e: Exception) -> ErrorCategory:
    msg = str(e).lower()
    if "memory" in msg or "oom" in msg or isinstance(e, MemoryError):
        return ErrorCategory.OUT_OF_MEMORY
    if "permission" in msg or "access" in msg or "denied" in msg:
        return ErrorCategory.WRITE_PERMISSION
    if "exif" in msg or "corrupt" in msg or "truncated" in msg:
        return ErrorCategory.EXIF_CORRUPT
    if "format" in msg or "decode" in msg or "identify" in msg:
        return ErrorCategory.UNSUPPORTED_FORMAT
    return ErrorCategory.UNKNOWN


def _extract_exif_orientation(image_path: Path) -> int:
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            orientation_tag = tags.get("Image Orientation", tags.get("EXIF Orientation"))
            if orientation_tag:
                return int(orientation_tag.values[0])
    except Exception:
        pass
    return 1


def _correct_orientation(pil_img: Image.Image, orientation: int) -> Image.Image:
    orientation_map = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }
    if orientation in orientation_map:
        return pil_img.transpose(orientation_map[orientation])
    return pil_img


def _detect_faces_hog(bgr_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h_orig, w_orig = bgr_img.shape[:2]
    max_dim = 800
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        small = cv2.resize(bgr_img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr_img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    try:
        rects, _ = hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(16, 16), scale=1.1
        )
        faces = []
        h_img, w_img = gray.shape[:2]
        for x, y, w, h in rects:
            if w * h > w_img * h_img * 0.01:
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                faces.append((x, y, x + w, y + h))
        return faces
    except Exception:
        return []


def _detect_saliency(bgr_img: np.ndarray) -> np.ndarray:
    h_orig, w_orig = bgr_img.shape[:2]
    max_dim = 500
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        small = cv2.resize(bgr_img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr_img

    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, sal_map = saliency.computeSaliency(small)
        small_sal = (sal_map * 255).astype(np.uint8)
    except Exception:
        small_sal = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        return cv2.resize(small_sal, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return small_sal


def _compute_entropy_map(gray_img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    h_orig, w_orig = gray_img.shape
    max_dim = 300
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        small = cv2.resize(gray_img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray_img

    lap = cv2.Laplacian(small, cv2.CV_32F)
    var_map = cv2.GaussianBlur(lap ** 2, (kernel_size, kernel_size), 0)
    norm = var_map.max()
    if norm > 0:
        ent_small = np.clip(var_map / norm * 255, 0, 255).astype(np.uint8)
    else:
        ent_small = np.zeros_like(small, dtype=np.uint8)

    if scale != 1.0:
        return cv2.resize(ent_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return ent_small


def _find_best_crop(
    bgr_img: np.ndarray,
    target_w: int,
    target_h: int,
    crop_cfg: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    h, w = bgr_img.shape[:2]
    aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > aspect:
        crop_h = h
        crop_w = int(h * aspect)
    else:
        crop_w = w
        crop_h = int(w / aspect)

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    if crop_w >= w and crop_h >= h:
        return 0, 0, w, h

    max_search_dim = 300
    scale = 1.0
    if max(h, w) > max_search_dim:
        scale = max_search_dim / max(h, w)
        small_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = bgr_img

    faces_small: List[Tuple[int, int, int, int]] = []
    if crop_cfg.get("protect_faces", True):
        faces_orig = _detect_faces_hog(bgr_img)
        faces_small = [
            (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
            for x1, y1, x2, y2 in faces_orig
        ]

    sh, sw = small_img.shape[:2]
    gray_s = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    sal_s = _detect_saliency(small_img)
    ent_w = crop_cfg.get("entropy_weight", 0.7)
    sal_w = crop_cfg.get("saliency_weight", 0.3)
    ent_s = _compute_entropy_map(gray_s, kernel_size=11)

    combined = cv2.addWeighted(
        ent_s.astype(np.float32) / 255.0, ent_w,
        sal_s.astype(np.float32) / 255.0, sal_w,
        0,
    )

    cw_s = int(crop_w * scale)
    ch_s = int(crop_h * scale)
    cw_s = max(10, min(cw_s, sw))
    ch_s = max(10, min(ch_s, sh))

    if cw_s >= sw and ch_s >= sh:
        return 0, 0, w, h

    pad_ratio = crop_cfg.get("padding_ratio", 0.1)
    step = max(2, min(sw - cw_s, sh - ch_s) // 25)
    if step < 1:
        step = 1

    integral = cv2.integral(combined)

    best_score = -1.0
    best_xs, best_ys = 0, 0

    h_range = list(range(0, sh - ch_s + 1, step))
    if not h_range or h_range[-1] != sh - ch_s:
        h_range.append(sh - ch_s)
    w_range = list(range(0, sw - cw_s + 1, step))
    if not w_range or w_range[-1] != sw - cw_s:
        w_range.append(sw - cw_s)

    for y in h_range:
        row_val = integral[y, :]
        row2_val = integral[y + ch_s, :]
        for x in w_range:
            sum_val = float(row2_val[x + cw_s] - row_val[x + cw_s] - row2_val[x] + row_val[x])
            area = cw_s * ch_s
            score = sum_val / area

            for fx1, fy1, fx2, fy2 in faces_small:
                cx1 = max(x, fx1)
                cy1 = max(y, fy1)
                cx2 = min(x + cw_s, fx2)
                cy2 = min(y + ch_s, fy2)
                if cx2 > cx1 and cy2 > cy1:
                    f_in = (cx2 - cx1) * (cy2 - cy1)
                    f_tot = (fx2 - fx1) * (fy2 - fy1)
                    if f_tot > 0:
                        score += 10.0 * (f_in / f_tot)

            if pad_ratio > 0:
                m = int(min(cw_s, ch_s) * pad_ratio)
                if cw_s > 2 * m and ch_s > 2 * m:
                    inner_sum = float(
                        integral[y + ch_s - m, x + cw_s - m]
                        - integral[y + m, x + cw_s - m]
                        - integral[y + ch_s - m, x + m]
                        + integral[y + m, x + m]
                    )
                    inner_area = (cw_s - 2 * m) * (ch_s - 2 * m)
                    score += (inner_sum / inner_area) * 0.3

            if score > best_score:
                best_score = score
                best_xs, best_ys = x, y

    orig_x = int(best_xs / scale)
    orig_y = int(best_ys / scale)
    orig_x = max(0, min(orig_x, w - crop_w))
    orig_y = max(0, min(orig_y, h - crop_h))
    return orig_x, orig_y, orig_x + crop_w, orig_y + crop_h


def _auto_white_balance(img: np.ndarray, wb_cfg: Dict[str, Any]) -> np.ndarray:
    method = wb_cfg.get("method", "auto")
    ref_white = wb_cfg.get("reference_white")
    threshold = wb_cfg.get("brightness_threshold", 240)

    if method == "reference" and ref_white is not None:
        ref = np.array(ref_white, dtype=np.float32)
        max_val = 255.0
        scale = max_val / np.clip(ref, 1, 255)
        return np.clip(img * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    brightest_mask = l > threshold
    if brightest_mask.sum() < 100:
        threshold = int(l.max() * 0.95)
        brightest_mask = l > threshold

    if brightest_mask.sum() >= 100:
        avg_a = float(np.mean(a[brightest_mask]))
        avg_b = float(np.mean(b[brightest_mask]))
        a = (a.astype(np.float32) - ((avg_a - 128) * (l.astype(np.float32) / 255.0))).clip(0, 255).astype(np.uint8)
        b = (b.astype(np.float32) - ((avg_b - 128) * (l.astype(np.float32) / 255.0))).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _generate_filename(
    template: str,
    sku_fields: Dict[str, str],
    index: int,
    padding: int = 4,
    suffix: str = "",
    extra: Optional[Dict[str, str]] = None,
) -> str:
    fields = {
        "brand": sku_fields.get("brand", "BRAND"),
        "category": sku_fields.get("category", "CAT"),
        "color": sku_fields.get("color", "CLR"),
        "index": str(index).zfill(padding),
        "sku": sku_fields.get("sku", ""),
    }
    if extra:
        fields.update(extra)
    name = template.format(**fields)
    return f"{name}{suffix}"


def _unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem, suffix = base_path.stem, base_path.suffix
    counter = 2
    while True:
        candidate = base_path.parent / f"{stem}_v{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _get_unique_output_path(
    output_dir: Path, filename: str, overwrite: bool
) -> Path:
    full_path = output_dir / filename
    if overwrite:
        return full_path
    return _unique_path(full_path)


def _extract_sku_from_path(
    rel_path: Path, default_fields: Dict[str, str]
) -> Dict[str, str]:
    fields = dict(default_fields)
    parts = rel_path.parent.parts
    if len(parts) >= 1:
        fields["category"] = parts[-1] if not fields.get("category") else fields["category"]
    if len(parts) >= 2:
        fields["brand"] = parts[-2] if not fields.get("brand") else fields["brand"]
    stem = rel_path.stem
    parts_sku = stem.replace("-", "_").replace(" ", "_").split("_")
    if len(parts_sku) >= 2 and not fields.get("color"):
        fields["color"] = parts_sku[-1]
    return fields


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _copy_exif(src_path: Path) -> Optional[bytes]:
    try:
        with Image.open(src_path) as src:
            exif_bytes = src.info.get("exif")
            if exif_bytes:
                return exif_bytes
    except Exception:
        pass
    return None


def process_single_image(
    task: ProcessTask,
    preset: Dict[str, Any],
    keep_structure: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ProcessResult:
    import time
    start = time.perf_counter()
    result = ProcessResult(input_path=task.input_path, task_index=task.task_id)

    try:
        if task.input_path.suffix.lower() not in SUPPORTED_INPUT:
            raise ValueError(f"不支持的文件格式: {task.input_path.suffix}")

        def _resolve_out_dir() -> Path:
            if task.sku_group_dir is not None:
                return task.sku_group_dir
            if keep_structure:
                return task.output_dir / task.rel_path.parent
            return task.output_dir

        if dry_run:
            sizes = preset.get("sizes", [])
            rename_cfg = preset.get("rename", {})
            sku = _extract_sku_from_path(task.rel_path, task.sku_fields)
            padding = rename_cfg.get("index_padding", 4)
            template = rename_cfg.get("template", "{brand}_{category}_{color}_{index}")
            out_fmt = preset.get("output", {}).get("format", "webp")
            out_quality = preset.get("output", {}).get("quality", 75)
            out_dir = _resolve_out_dir()
            extra_fields = None
            name_index = task.index
            if task.is_sample and task.sample_position:
                extra_fields = {"index": task.sample_position}
                name_index = 0
            for sz in sizes:
                fname = _generate_filename(
                    template, sku, name_index, padding, sz.get("suffix", ""), extra_fields
                ) + f".{out_fmt}"
                result.output_paths.append(out_dir / fname)
                result.dry_run_details.append({
                    "size_name": sz.get("name", ""),
                    "width": sz["width"],
                    "height": sz["height"],
                    "suffix": sz.get("suffix", ""),
                    "filename": fname,
                    "format": out_fmt,
                    "quality": out_quality,
                    "output_dir": str(out_dir),
                })
            if task.extra_sizes:
                for sz in task.extra_sizes:
                    fname = _generate_filename(
                        template, sku, name_index, padding, sz.get("suffix", ""), extra_fields
                    ) + f".{out_fmt}"
                    result.output_paths.append(out_dir / fname)
                    result.dry_run_details.append({
                        "size_name": sz.get("name", "extra"),
                        "width": sz["width"],
                        "height": sz["height"],
                        "suffix": sz.get("suffix", ""),
                        "filename": fname,
                        "format": out_fmt,
                        "quality": out_quality,
                        "output_dir": str(out_dir),
                    })
            result.success = True
            result.skipped = True
            result.skip_reason = "dry-run"
            result.duration_ms = int((time.perf_counter() - start) * 1000)
            return result

        orientation = _extract_exif_orientation(task.input_path)
        with Image.open(task.input_path) as pil_img:
            pil_img = _correct_orientation(pil_img, orientation)
            bgr_img = pil_to_cv2(pil_img)

        crop_cfg = preset.get("crop", {})
        wb_cfg = preset.get("white_balance", {})
        out_cfg = preset.get("output", {})
        rename_cfg = preset.get("rename", {})
        sizes = preset.get("sizes", [])

        bgr_balanced = _auto_white_balance(bgr_img, wb_cfg)

        exif_bytes = _copy_exif(task.input_path) if out_cfg.get("keep_exif", True) else None

        sku = _extract_sku_from_path(task.rel_path, task.sku_fields)
        padding = rename_cfg.get("index_padding", 4)
        template = rename_cfg.get("template", "{brand}_{category}_{color}_{index}")
        out_fmt = out_cfg.get("format", "webp").lower()
        quality = out_cfg.get("quality", 75)

        base_out_dir = task.sku_group_dir if task.sku_group_dir is not None else (
            task.output_dir / task.rel_path.parent if keep_structure else task.output_dir
        )
        base_out_dir.mkdir(parents=True, exist_ok=True)

        sample_extra_fields = None
        sample_name_index = task.index
        if task.is_sample and task.sample_position:
            sample_extra_fields = {"index": task.sample_position}
            sample_name_index = 0

        all_sizes = list(sizes)
        if task.extra_sizes:
            all_sizes.extend(task.extra_sizes)

        for sz in all_sizes:
            tw, th = sz["width"], sz["height"]
            cx1, cy1, cx2, cy2 = _find_best_crop(bgr_balanced, tw, th, crop_cfg)
            cropped = bgr_balanced[cy1:cy2, cx1:cx2]
            resized = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)
            pil_result = cv2_to_pil(resized)

            fname = _generate_filename(
                template, sku, sample_name_index, padding, sz.get("suffix", ""), sample_extra_fields
            ) + f".{out_fmt}"
            out_path = _get_unique_output_path(base_out_dir, fname, overwrite)

            save_kwargs: Dict[str, Any] = {"quality": quality}
            if out_fmt == "webp":
                save_kwargs["method"] = 6
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
            elif exif_bytes:
                save_kwargs["exif"] = exif_bytes

            pil_result.save(out_path, **save_kwargs)
            result.output_paths.append(out_path)

        result.success = True
        result.duration_ms = int((time.perf_counter() - start) * 1000)

    except Exception as e:
        result.success = False
        result.duration_ms = int((time.perf_counter() - start) * 1000)
        category = _classify_error(e)
        result.error = ProcessError(
            path=task.input_path,
            category=category,
            message=str(e),
            stacktrace=traceback.format_exc(),
        )

    return result


def _worker_wrapper(args: Tuple[ProcessTask, Dict, bool, bool, bool, int]) -> ProcessResult:
    task, preset, keep_structure, overwrite, dry_run, task_idx = args
    try:
        return process_single_image(task, preset, keep_structure, overwrite, dry_run)
    except Exception as e:
        return ProcessResult(
            input_path=task.input_path,
            task_index=task_idx,
            success=False,
            error=ProcessError(
                path=task.input_path,
                category=_classify_error(e),
                message=str(e),
                stacktrace=traceback.format_exc(),
            ),
        )


def scan_input_directory(
    input_dir: Path, recursive: bool = True
) -> List[Tuple[Path, Path]]:
    results: List[Tuple[Path, Path]] = []
    if recursive:
        for root, _, files in sorted(os.walk(input_dir)):
            for fname in sorted(files):
                p = Path(root) / fname
                if p.suffix.lower() in SUPPORTED_INPUT:
                    rel = p.relative_to(input_dir)
                    results.append((p, rel))
    else:
        for fname in sorted(os.listdir(input_dir)):
            p = input_dir / fname
            if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT:
                results.append((p, Path(fname)))
    return results


def write_errors_log(errors: List[ProcessError], log_path: Path) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for err in errors:
            f.write(f"{'='*60}\n")
            f.write(f"文件: {err.path}\n")
            f.write(f"分类: {err.category.value}\n")
            f.write(f"错误: {err.message}\n")
            f.write(f"堆栈:\n{err.stacktrace}\n\n")


def _compute_sku_key(task: ProcessTask) -> str:
    b = task.sku_fields.get("brand", "")
    c = task.sku_fields.get("category", "")
    cl = task.sku_fields.get("color", "")
    parts = [p for p in [b, c, cl] if p]
    if parts:
        return "/".join(parts)
    parts = list(task.rel_path.parent.parts)
    if parts:
        return "/".join(parts)
    return "default"


def insert_sample_markers(
    tasks: List[ProcessTask],
    sample_cfg: Dict[str, Any],
    sample_image_path: Optional[Path] = None,
    sku_boundary: bool = True,
) -> List[ProcessTask]:
    interval = sample_cfg.get("sample_interval", 20)
    if not sample_image_path:
        return tasks
    if interval <= 0 and not sku_boundary:
        return tasks

    sku_groups: Dict[str, List[int]] = {}
    for i, task in enumerate(tasks):
        key = _compute_sku_key(task)
        if key not in sku_groups:
            sku_groups[key] = []
        sku_groups[key].append(i)

    sku_boundary_indices: Dict[int, str] = {}
    if sku_boundary:
        for key, indices in sku_groups.items():
            if indices:
                sku_boundary_indices[indices[0]] = "sku_start"
                if indices[-1] != indices[0]:
                    sku_boundary_indices[indices[-1]] = "sku_end"

    result: List[ProcessTask] = []
    for i, task in enumerate(tasks):
        if i in sku_boundary_indices:
            pos = sku_boundary_indices[i]
            if pos == "sku_start":
                result.append(_make_sample_task(
                    sample_image_path, task.output_dir,
                    task.sku_group_dir, pos, task.sku_fields,
                ))

        result.append(task)

        if i in sku_boundary_indices:
            pos = sku_boundary_indices[i]
            if pos == "sku_end":
                result.append(_make_sample_task(
                    sample_image_path, task.output_dir,
                    task.sku_group_dir, pos, task.sku_fields,
                ))

        if interval > 0 and (i + 1) % interval == 0 and i < len(tasks) - 1:
            result.append(_make_sample_task(
                sample_image_path, task.output_dir,
                task.sku_group_dir, f"interval_{i+1}", task.sku_fields,
            ))

    return result


def _make_sample_task(
    sample_image_path: Path,
    output_dir: Path,
    sku_group_dir: Optional[Path],
    position: str,
    sku_fields: Optional[Dict[str, str]] = None,
) -> ProcessTask:
    return ProcessTask(
        input_path=sample_image_path,
        rel_path=Path("_SAMPLE.webp"),
        output_dir=output_dir,
        index=0,
        sku_fields=dict(sku_fields) if sku_fields else {"brand": "SAMPLE", "category": "CHECK", "color": "REF"},
        extra_sizes=[{"name": "sample", "width": 800, "height": 800, "suffix": ""}],
        sku_group_dir=sku_group_dir,
        is_sample=True,
        sample_position=position,
    )


def batch_process(
    tasks: List[ProcessTask],
    preset: Dict[str, Any],
    workers: int = 4,
    keep_structure: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[ProcessResult, int, int], None]] = None,
) -> BatchStats:
    import os
    import concurrent.futures
    import time

    stats = BatchStats(total=len(tasks))
    if not tasks:
        return stats

    start_time = time.perf_counter()

    if workers <= 1 or len(tasks) <= 2 or dry_run:
        for idx, task in enumerate(tasks):
            task.task_id = idx
            result = process_single_image(task, preset, keep_structure, overwrite, dry_run)
            _update_stats(stats, result)
            if progress_callback:
                progress_callback(result, idx + 1, len(tasks))
    else:
        for idx, task in enumerate(tasks):
            task.task_id = idx
        worker_args = [
            (task, preset, keep_structure, overwrite, dry_run, i) for i, task in enumerate(tasks)
        ]
        completed = 0
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(_worker_wrapper, arg): i
                    for i, arg in enumerate(worker_args)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        idx = future_to_idx[future]
                        result = ProcessResult(
                            input_path=tasks[idx].input_path,
                            task_index=idx,
                            success=False,
                            error=ProcessError(
                                path=tasks[idx].input_path,
                                category=_classify_error(e),
                                message=str(e),
                                stacktrace=traceback.format_exc(),
                            ),
                        )
                    _update_stats(stats, result)
                    if progress_callback:
                        progress_callback(result, completed, len(tasks))
        except Exception as e:
            for i in range(completed, len(tasks)):
                result = ProcessResult(
                    input_path=tasks[i].input_path,
                    task_index=i,
                    success=False,
                    error=ProcessError(
                        path=tasks[i].input_path,
                        category=_classify_error(e),
                        message=f"进程池异常: {e}",
                        stacktrace=traceback.format_exc(),
                    ),
                )
                _update_stats(stats, result)
                completed += 1
                if progress_callback:
                    progress_callback(result, completed, len(tasks))

    stats.duration_ms = int((time.perf_counter() - start_time) * 1000)
    return stats


def _update_stats(stats: BatchStats, result: ProcessResult) -> None:
    if result.skipped:
        stats.skipped += 1
        stats.success += 1
    elif result.success:
        stats.success += 1
    else:
        stats.failed += 1
        if result.error:
            stats.add_error(result.error)


def read_exif_tags(image_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)
            for tag, value in tags.items():
                result[str(tag)] = str(value)
    except Exception as e:
        result["ERROR"] = str(e)
    return result


import os
