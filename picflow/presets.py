"""预设配置模块 - 平台尺寸、白平衡参数、输出格式、命名模板"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR = Path.home() / ".picflow"
PRESETS_FILE = CONFIG_DIR / "presets.json"
CONFIG_FILE = CONFIG_DIR / "config.json"

BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "taobao": {
        "name": "淘宝/天猫",
        "description": "淘宝天猫主图及详情页标准",
        "output": {
            "format": "webp",
            "quality": 75,
            "keep_exif": True,
        },
        "sizes": [
            {"name": "主图", "width": 800, "height": 800, "suffix": "_main"},
            {"name": "详情", "width": 750, "height": 1000, "suffix": "_detail"},
        ],
        "white_balance": {
            "method": "auto",
            "reference_white": None,
            "brightness_threshold": 240,
        },
        "crop": {
            "mode": "saliency",
            "padding_ratio": 0.1,
            "protect_faces": True,
            "entropy_weight": 0.7,
            "saliency_weight": 0.3,
        },
        "rename": {
            "template": "{brand}_{category}_{color}_{index}",
            "index_padding": 4,
            "start_index": 1,
        },
        "quality_check": {
            "sample_interval": 20,
            "sample_filename": "_SAMPLE_CHECK.webp",
        },
    },
    "jd": {
        "name": "京东",
        "description": "京东主图及详情页标准",
        "output": {
            "format": "webp",
            "quality": 78,
            "keep_exif": True,
        },
        "sizes": [
            {"name": "主图", "width": 800, "height": 800, "suffix": "_main"},
            {"name": "PC详情", "width": 990, "height": 1500, "suffix": "_pc"},
            {"name": "移动端", "width": 750, "height": 1200, "suffix": "_m"},
        ],
        "white_balance": {
            "method": "auto",
            "reference_white": [245, 245, 245],
            "brightness_threshold": 235,
        },
        "crop": {
            "mode": "saliency",
            "padding_ratio": 0.08,
            "protect_faces": True,
            "entropy_weight": 0.6,
            "saliency_weight": 0.4,
        },
        "rename": {
            "template": "JD_{brand}_{category}_{color}_{index}",
            "index_padding": 4,
            "start_index": 1,
        },
        "quality_check": {
            "sample_interval": 25,
            "sample_filename": "_JD_SAMPLE.webp",
        },
    },
    "independent": {
        "name": "独立站",
        "description": "独立站/Shopify多尺寸适配",
        "output": {
            "format": "webp",
            "quality": 80,
            "keep_exif": True,
        },
        "sizes": [
            {"name": "高清大图", "width": 1200, "height": 1200, "suffix": "_xl"},
            {"name": "产品图", "width": 800, "height": 800, "suffix": "_lg"},
            {"name": "缩略图", "width": 400, "height": 400, "suffix": "_th"},
            {"name": "Banner", "width": 1600, "height": 600, "suffix": "_bn"},
        ],
        "white_balance": {
            "method": "reference",
            "reference_white": [250, 250, 250],
            "brightness_threshold": 245,
        },
        "crop": {
            "mode": "entropy",
            "padding_ratio": 0.12,
            "protect_faces": True,
            "entropy_weight": 0.8,
            "saliency_weight": 0.2,
        },
        "rename": {
            "template": "{brand}-{category}-{color}-{index}",
            "index_padding": 4,
            "start_index": 1,
        },
        "quality_check": {
            "sample_interval": 15,
            "sample_filename": "_SAMPLE_INDEPENDENT.webp",
        },
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_preset": "taobao",
    "workers": 4,
    "hash_threshold": 8,
    "dry_run": False,
    "error_log": "errors.log",
    "keep_original_structure": True,
    "overwrite": False,
    "sample_image_path": None,
    "sample_interval": None,
    "logging": {
        "level": "INFO",
        "to_console": True,
        "to_file": True,
    },
}


class PresetManager:
    """预设管理器 - 加载、保存、管理平台预设"""

    def __init__(self) -> None:
        self._ensure_config_dir()
        self.user_presets: Dict[str, Dict[str, Any]] = {}
        self.config: Dict[str, Any] = {}
        self._load_user_presets()
        self._load_config()

    @staticmethod
    def _ensure_config_dir() -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_user_presets(self) -> None:
        if PRESETS_FILE.exists():
            try:
                with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                    self.user_presets = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.user_presets = {}

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.config = deepcopy(DEFAULT_CONFIG)
        else:
            self.config = deepcopy(DEFAULT_CONFIG)

    def save_config(self) -> None:
        self._ensure_config_dir()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def save_user_presets(self) -> None:
        self._ensure_config_dir()
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.user_presets, f, indent=2, ensure_ascii=False)

    def init_config(self, force: bool = False) -> bool:
        if CONFIG_FILE.exists() and not force:
            return False
        self.config = deepcopy(DEFAULT_CONFIG)
        self.save_config()
        return True

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self.user_presets:
            return deepcopy(self.user_presets[name])
        if name in BUILTIN_PRESETS:
            return deepcopy(BUILTIN_PRESETS[name])
        return None

    def list_presets(self) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        for name, preset in BUILTIN_PRESETS.items():
            result.append({
                "name": name,
                "display_name": preset["name"],
                "description": preset["description"],
                "source": "builtin",
            })
        for name, preset in self.user_presets.items():
            result.append({
                "name": name,
                "display_name": preset.get("name", name),
                "description": preset.get("description", ""),
                "source": "user",
            })
        return result

    def add_preset(self, name: str, preset: Dict[str, Any]) -> None:
        if name in BUILTIN_PRESETS:
            raise ValueError(f"预设名称 '{name}' 已被内置预设占用")
        self.user_presets[name] = preset
        self.save_user_presets()

    def remove_preset(self, name: str) -> bool:
        if name in self.user_presets:
            del self.user_presets[name]
            self.save_user_presets()
            return True
        return False

    def export_preset(self, name: str, export_path: Path) -> bool:
        preset = self.get_preset(name)
        if preset is None:
            return False
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({name: preset}, f, indent=2, ensure_ascii=False)
        return True

    def import_presets(self, import_path: Path) -> List[str]:
        with open(import_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        imported: List[str] = []
        for name, preset in data.items():
            if name not in BUILTIN_PRESETS:
                self.user_presets[name] = preset
                imported.append(name)
        if imported:
            self.save_user_presets()
        return imported

    def get_all_presets(self) -> Dict[str, Dict[str, Any]]:
        all_presets = deepcopy(BUILTIN_PRESETS)
        all_presets.update(deepcopy(self.user_presets))
        return all_presets

    def set_config(self, key: str, value: Any) -> None:
        keys = key.split(".")
        cfg = self.config
        for k in keys[:-1]:
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value
        self.save_config()

    def get_config(self, key: Optional[str] = None) -> Any:
        if key is None:
            return deepcopy(self.config)
        keys = key.split(".")
        cfg = self.config
        for k in keys:
            if isinstance(cfg, dict) and k in cfg:
                cfg = cfg[k]
            else:
                return None
        return deepcopy(cfg)
