"""PicFlow CLI入口 - 基于Typer构建的命令行界面

验收命令示例:
  1. hash直接入口(非交互):  picflow hash ./images --action link --yes
  2. hash-scan子命令:       picflow hash-scan scan ./images -t 10 -a report
  3. 自定义尺寸处理:        picflow process ./raw --preset taobao --width 600 --height 600 --quality 85 -o ./done
  4. 白平衡参考值覆盖:      picflow process ./raw --preset jd --white-reference 245,245,245
  5. 自动抽检配置:          picflow config set sample_image_path ./ref.jpg
                             picflow config set sample_interval 10
                             picflow process ./raw --preset taobao -o ./done
  6. 4进程并行处理:         picflow process ./raw --preset taobao --workers 4 -o ./done --skip-hash
  7. dry-run预览:           picflow process ./raw --preset independent --dry-run
  8. SKU分组输出:           picflow process ./raw --preset taobao --sku-group -o ./done
  9. 增量处理(跳过未变):    picflow process ./raw --preset taobao --incremental -o ./done
  10. 处理报告:             picflow report ./done
  11. 报告导出CSV:          picflow report ./done --csv report.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from .hasher import DuplicateGroup, ImageHasher, SUPPORTED_EXTENSIONS
from .manifest import Manifest, ManifestEntry
from .presets import CONFIG_FILE, PRESETS_FILE, PresetManager
from .processor import (
    BatchStats,
    ErrorCategory,
    ProcessResult,
    ProcessTask,
    batch_process,
    read_exif_tags,
    scan_input_directory,
    write_errors_log,
    _update_stats,
    _compute_sku_key,
)

app = typer.Typer(
    name="picflow",
    help="电商商品图片批量处理工具 - 智能裁剪、统一调色、WebP转换、自动重命名",
    add_completion=False,
    no_args_is_help=True,
)

console = Console(force_terminal=True)
err_console = Console(stderr=True, force_terminal=True)


def _banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]PicFlow[/bold cyan] - 电商图片批量处理工具\n"
            "[dim]智能裁剪 · 统一调色 · WebP转换 · 自动重命名 · 去重检测[/dim]",
            border_style="cyan",
        )
    )


@app.command("init")
def cmd_init(
    force: bool = typer.Option(False, "--force", "-f", help="覆盖已有配置"),
) -> None:
    """初始化 ~/.picflow 配置目录与默认配置"""
    _banner()
    pm = PresetManager()
    created = pm.init_config(force=force)
    if created:
        console.print(f"[green]✓[/green] 配置已初始化: {CONFIG_FILE}")
        console.print(f"[green]✓[/green] 预设目录: {PRESETS_FILE.parent}")
    else:
        console.print("[yellow]![/yellow] 配置已存在，使用 --force 覆盖")
    raise typer.Exit(0)


@app.command("hash")
def cmd_hash(
    directory: str = typer.Argument(..., help="要扫描的图片目录，或 'scan' 关键字（老写法：picflow hash scan ./images）"),
    dir2: Optional[Path] = typer.Argument(None, exists=True, file_okay=False, dir_okay=True, help="当第一个参数是 'scan' 时，这里是真正的目录"),
    threshold: int = typer.Option(8, "--threshold", "-t", help="汉明距离阈值（越小越严格）"),
    action: str = typer.Option("report", "--action", "-a", help="report: 只报告 | link: 软链接归集到 dupes/ | move: 复制归集"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="归集输出目录（默认与输入同）"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    yes: Optional[bool] = typer.Option(None, "--yes/-y", "--no-yes", help="跳过交互确认，--action link/move 时默认自动确认"),
) -> None:
    """扫描目录检测重复图片（直接入口：picflow hash ./images  或  老写法：picflow hash scan ./images）"""
    _banner()
    if directory == "scan":
        if dir2 is None:
            err_console.print("[red]✗[/red] 老写法 picflow hash scan <目录> 需要指定目录参数")
            raise typer.Exit(1)
        real_dir = dir2
    else:
        real_dir = Path(directory)
        if not real_dir.exists() or not real_dir.is_dir():
            err_console.print(f"[red]✗[/red] 目录不存在: {real_dir}")
            raise typer.Exit(1)
    auto_yes = yes if yes is not None else (action != "report")
    _hash_scan_impl(real_dir, threshold, action, output, recursive, auto_yes)


hash_app = typer.Typer(help="图片去重检测（子命令组）")
app.add_typer(hash_app, name="hash-scan")


@hash_app.callback(invoke_without_command=True)
def hash_scan_main(
    ctx: typer.Context,
) -> None:
    if ctx.invoked_subcommand is None:
        console.print("[dim]使用 picflow hash-scan scan <目录> 或 picflow hash <目录>[/dim]")


@hash_app.command("scan")
def hash_scan(
    directory: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    threshold: int = typer.Option(8, "--threshold", "-t", help="汉明距离阈值（越小越严格）"),
    action: str = typer.Option("report", "--action", "-a", help="report: 只报告 | link: 软链接归集到 dupes/ | move: 复制归集"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="归集输出目录（默认与输入同）"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    yes: Optional[bool] = typer.Option(None, "--yes/-y", "--no-yes", help="跳过交互确认，--action link/move 时默认自动确认"),
) -> None:
    """扫描目录检测重复图片"""
    _banner()
    auto_yes = yes if yes is not None else (action != "report")
    _hash_scan_impl(directory, threshold, action, output, recursive, auto_yes)


config_app = typer.Typer(help="配置管理")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """初始化配置文件"""
    cmd_init(force=force)


@config_app.command("show")
def config_show() -> None:
    """显示当前所有配置参数"""
    pm = PresetManager()
    cfg = pm.get_config()

    table = Table(title="当前配置", show_lines=False, header_style="cyan")
    table.add_column("参数", style="bold")
    table.add_column("值")

    def _add_rows(data: Dict[str, Any], prefix: str = "") -> None:
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _add_rows(v, key)
            else:
                table.add_row(key, str(v))

    _add_rows(cfg)
    console.print(table)

    preset_table = Table(title="可用 Presets", show_lines=False, header_style="magenta")
    preset_table.add_column("名称", style="bold")
    preset_table.add_column("显示名")
    preset_table.add_column("来源")
    preset_table.add_column("描述")
    for p in pm.list_presets():
        src_style = "green" if p["source"] == "builtin" else "yellow"
        preset_table.add_row(
            p["name"], p["display_name"], f"[{src_style}]{p['source']}[/{src_style}]", p["description"]
        )
    console.print(preset_table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="配置键，例如 workers 或 logging.level"),
    value: str = typer.Argument(..., help="配置值"),
) -> None:
    """设置配置项"""
    pm = PresetManager()
    try:
        parsed: Any = value
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        elif value.isdigit():
            parsed = int(value)
        elif value.replace(".", "", 1).isdigit():
            parsed = float(value)
        pm.set_config(key, parsed)
        console.print(f"[green]✓[/green] {key} = {parsed}")
    except Exception as e:
        err_console.print(f"[red]✗[/red] 设置失败: {e}")
        raise typer.Exit(1)


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="配置键"),
) -> None:
    """读取配置项"""
    pm = PresetManager()
    val = pm.get_config(key)
    if val is None:
        err_console.print(f"[red]✗[/red] 键不存在: {key}")
        raise typer.Exit(1)
    if isinstance(val, (dict, list)):
        console.print_json(json.dumps(val, ensure_ascii=False, indent=2))
    else:
        console.print(f"{key} = [cyan]{val}[/cyan]")


@config_app.command("preset-add")
def config_preset_add(
    name: str = typer.Argument(..., help="Preset名称"),
    json_file: Path = typer.Argument(..., exists=True, readable=True, help="JSON配置文件路径"),
) -> None:
    """从JSON文件添加自定义preset"""
    pm = PresetManager()
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            preset = json.load(f)
        pm.add_preset(name, preset)
        console.print(f"[green]✓[/green] Preset '{name}' 已添加")
    except Exception as e:
        err_console.print(f"[red]✗[/red] 添加失败: {e}")
        raise typer.Exit(1)


@config_app.command("preset-remove")
def config_preset_remove(
    name: str = typer.Argument(..., help="Preset名称"),
) -> None:
    """删除自定义preset"""
    pm = PresetManager()
    if pm.remove_preset(name):
        console.print(f"[green]✓[/green] Preset '{name}' 已删除")
    else:
        err_console.print(f"[red]✗[/red] Preset '{name}' 不存在或为内置预设")
        raise typer.Exit(1)


@config_app.command("preset-export")
def config_preset_export(
    name: str = typer.Argument(..., help="Preset名称"),
    output: Path = typer.Option(Path("preset_export.json"), "--output", "-o", help="导出路径"),
) -> None:
    """导出preset为JSON文件（可分享）"""
    pm = PresetManager()
    if pm.export_preset(name, output):
        console.print(f"[green]✓[/green] 已导出到: {output}")
    else:
        err_console.print(f"[red]✗[/red] Preset '{name}' 不存在")
        raise typer.Exit(1)


@config_app.command("preset-import")
def config_preset_import(
    input: Path = typer.Argument(..., exists=True, readable=True, help="导入文件路径"),
) -> None:
    """从JSON文件导入preset"""
    pm = PresetManager()
    imported = pm.import_presets(input)
    if imported:
        console.print(f"[green]✓[/green] 成功导入 {len(imported)} 个 preset: {', '.join(imported)}")
    else:
        console.print("[yellow]![/yellow] 没有导入新的 preset（可能名称冲突或文件为空）")


@app.command("process")
def cmd_process(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="输入图片目录"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录（默认在输入目录下创建_processed）"),
    preset: str = typer.Option(..., "--preset", "-p", help="处理预设: taobao/jd/independent 或自定义"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="并行进程数"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="只打印操作，不实际处理"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="递归扫描子目录"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的文件"),
    skip_hash: bool = typer.Option(False, "--skip-hash", help="跳过重复检测"),
    brand: Optional[str] = typer.Option(None, "--brand", help="SKU品牌字段（默认从目录识别）"),
    category: Optional[str] = typer.Option(None, "--category", help="SKU品类字段（默认从目录识别）"),
    color: Optional[str] = typer.Option(None, "--color", help="SKU颜色字段（默认从文件名识别）"),
    sample_image: Optional[Path] = typer.Option(None, "--sample", help="抽检样图路径（覆盖config中的默认值）"),
    hash_threshold: Optional[int] = typer.Option(None, "--hash-threshold", help="汉明距离阈值"),
    override_width: Optional[int] = typer.Option(None, "--width", help="覆盖preset的输出宽度"),
    override_height: Optional[int] = typer.Option(None, "--height", help="覆盖preset的输出高度"),
    override_quality: Optional[int] = typer.Option(None, "--quality", help="覆盖preset的输出质量"),
    override_format: Optional[str] = typer.Option(None, "--format", help="覆盖preset的输出格式(如webp/jpg/png)"),
    override_white_ref: Optional[str] = typer.Option(None, "--white-reference", help="覆盖白平衡参考值，如 245,245,245"),
    sku_group: bool = typer.Option(False, "--sku-group", help="按SKU分组到品牌/品类/颜色子目录"),
    incremental: bool = typer.Option(False, "--incremental", help="增量处理：跳过未变化的已有输出"),
    no_sku_boundary: bool = typer.Option(False, "--no-sku-boundary", help="不在每个SKU首尾插入抽检参考图"),
) -> None:
    """批量处理图片（核心命令）"""
    _banner()

    pm = PresetManager()
    preset_cfg = pm.get_preset(preset)
    if preset_cfg is None:
        err_console.print(f"[red]✗[/red] Preset '{preset}' 不存在。可用: {[p['name'] for p in pm.list_presets()]}")
        raise typer.Exit(1)

    global_cfg = pm.get_config()
    workers = workers or global_cfg.get("workers", 4)
    hash_threshold = hash_threshold or global_cfg.get("hash_threshold", 8)

    if override_quality is not None:
        preset_cfg["output"]["quality"] = override_quality
    if override_format is not None:
        preset_cfg["output"]["format"] = override_format.lower()
    if override_width is not None or override_height is not None:
        sizes = preset_cfg.get("sizes", [])
        if override_width is not None and override_height is not None:
            preset_cfg["sizes"] = [
                {"name": "自定义", "width": override_width, "height": override_height, "suffix": s.get("suffix", "")}
                for s in (sizes or [{"suffix": ""}])
            ] if len(sizes) <= 1 else [
                {"name": "自定义", "width": override_width, "height": override_height, "suffix": sizes[0].get("suffix", "")}
            ]
        elif sizes:
            preset_cfg["sizes"] = [
                {**s,
                 "width": override_width if override_width is not None else s["width"],
                 "height": override_height if override_height is not None else s["height"]}
                for s in sizes
            ]
    if override_white_ref is not None:
        try:
            vals = [int(v.strip()) for v in override_white_ref.split(",")]
            if len(vals) == 3:
                preset_cfg["white_balance"]["method"] = "reference"
                preset_cfg["white_balance"]["reference_white"] = vals
            else:
                err_console.print("[red]✗[/red] --white-reference 格式: R,G,B 如 245,245,245")
                raise typer.Exit(1)
        except ValueError:
            err_console.print("[red]✗[/red] --white-reference 格式: R,G,B 如 245,245,245")
            raise typer.Exit(1)

    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]→[/cyan] 输入目录: {input_dir}")
    console.print(f"[cyan]→[/cyan] 输出目录: {output_dir}")
    console.print(f"[cyan]→[/cyan] 预设: [bold]{preset_cfg.get('name', preset)}[/bold]")
    console.print(f"[cyan]→[/cyan] 并行进程: {workers}")
    if sku_group:
        console.print(f"[cyan]→[/cyan] SKU分组: [bold]开启[/bold]（按品牌/品类/颜色子目录）")
    if incremental:
        console.print(f"[cyan]→[/cyan] 增量处理: [bold]开启[/bold]（跳过未变化文件）")

    final_sizes = preset_cfg.get("sizes", [])
    sizes_str = " / ".join(f'{s.get("name","")}{s["width"]}×{s["height"]}' for s in final_sizes)
    console.print(f"[cyan]→[/cyan] 输出尺寸: {sizes_str}")
    console.print(f"[cyan]→[/cyan] 输出格式: {preset_cfg.get('output',{}).get('format','webp')} q={preset_cfg.get('output',{}).get('quality',75)}")

    wb_method = preset_cfg.get("white_balance", {}).get("method", "auto")
    wb_ref = preset_cfg.get("white_balance", {}).get("reference_white")
    wb_str = f"{wb_method}" + (f" ref={wb_ref}" if wb_ref else "")
    console.print(f"[cyan]→[/cyan] 白平衡: {wb_str}")

    if dry_run:
        console.print("[yellow]⚠ Dry-run 模式，不实际处理文件[/yellow]")

    scanned = scan_input_directory(input_dir, recursive=recursive)
    if not scanned:
        err_console.print(f"[red]✗[/red] 未找到支持的图片文件（支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}）")
        raise typer.Exit(1)

    console.print(f"[cyan]→[/cyan] 发现图片: [bold]{len(scanned)}[/bold] 张")

    skip_paths: set = set()
    if not skip_hash:
        hasher = ImageHasher(threshold=hash_threshold)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("计算感知哈希...", total=len(scanned))

            def _hash_cb(idx: int, total: int, path: Path) -> None:
                progress.update(task_id, advance=1, description=f"计算哈希: {path.name[:40]}")

            hash_infos = hasher.scan_directory(
                input_dir, recursive=recursive, progress_callback=_hash_cb
            )

        console.print(f"[cyan]→[/cyan] 哈希计算完成: {len(hash_infos)} 张有效")

        dup_pairs = []
        if len(hash_infos) > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("检测重复...", total=len(hash_infos))

                def _dup_cb(idx: int, total: int) -> None:
                    progress.update(task_id, completed=idx)

                dup_pairs = hasher.find_duplicates(hash_infos, progress_callback=_dup_cb)

        if dup_pairs:
            groups = hasher.group_duplicates(dup_pairs)
            console.print(f"[yellow]⚠[/yellow] 发现 [bold]{len(groups)}[/bold] 组重复图片，共 [bold]{len(dup_pairs)}[/bold] 对")

            table = Table(title="重复分组（推荐保留）", show_lines=False, header_style="yellow")
            table.add_column("#", style="bold", justify="right")
            table.add_column("推荐保留", style="green")
            table.add_column("重复数")
            table.add_column("平均距离")
            for i, g in enumerate(groups[:20], 1):
                avg_d = int(sum(g.distances) / len(g.distances)) if g.distances else 0
                table.add_row(
                    str(i),
                    f"{g.images[g.keep_index].path.name} ({g.images[g.keep_index].resolution[0]}x{g.images[g.keep_index].resolution[1]})",
                    str(len(g.images)),
                    str(avg_d),
                )
            if len(groups) > 20:
                table.add_row("...", f"(还有 {len(groups) - 20} 组)", "", "")
            console.print(table)

            skip = Confirm.ask("是否[bold]跳过[/bold]重复项，只处理推荐保留的图片?", default=True)
            if skip:
                skip_paths = hasher.get_skip_set(dup_pairs)
                console.print(f"[green]✓[/green] 将跳过 [bold]{len(skip_paths)}[/bold] 张重复图片")
            else:
                console.print("[yellow]![/yellow] 将处理所有图片（包括重复项）")

    incremental_skip: set = set()
    if incremental and not dry_run:
        old_manifest = Manifest.load(output_dir)
        if old_manifest is not None:
            old_manifest.build_skip_cache()
            before = len(scanned)
            scanned = [(p, r) for p, r in scanned if not old_manifest.should_skip(p)]
            incremental_skip_count = before - len(scanned)
            if incremental_skip_count > 0:
                console.print(f"[cyan]→[/cyan] 增量跳过: [bold]{incremental_skip_count}[/bold] 张未变化文件")

    tasks: List[ProcessTask] = []
    idx = preset_cfg.get("rename", {}).get("start_index", 1)
    for abs_path, rel_path in scanned:
        if abs_path in skip_paths:
            continue
        task_sku_fields: Dict[str, str] = {}
        if brand:
            task_sku_fields["brand"] = brand
        if category:
            task_sku_fields["category"] = category
        if color:
            task_sku_fields["color"] = color
        sku_group_dir = None
        if sku_group:
            from .processor import _extract_sku_from_path
            sku_resolved = _extract_sku_from_path(rel_path, task_sku_fields)
            task_sku_fields = sku_resolved
            sku_key_parts = [sku_resolved.get("brand", ""), sku_resolved.get("category", ""), sku_resolved.get("color", "")]
            sku_key_parts = [p for p in sku_key_parts if p]
            if sku_key_parts:
                sku_group_dir = output_dir / Path(*sku_key_parts)
            else:
                sku_group_dir = output_dir / rel_path.parent

        task = ProcessTask(
            input_path=abs_path,
            rel_path=rel_path,
            output_dir=output_dir,
            index=idx,
            sku_fields=task_sku_fields,
            sku_group_dir=sku_group_dir,
        )
        tasks.append(task)
        idx += 1

    if not tasks:
        console.print("[yellow]![/yellow] 没有需要处理的图片")
        raise typer.Exit(0)

    effective_sample_path = sample_image
    if effective_sample_path is None:
        cfg_sample = global_cfg.get("sample_image_path")
        if cfg_sample:
            p = Path(cfg_sample)
            if p.exists():
                effective_sample_path = p

    sample_cfg = preset_cfg.get("quality_check", {})
    effective_interval = sample_cfg.get("sample_interval", 20)
    cfg_interval = global_cfg.get("sample_interval")
    if cfg_interval is not None:
        effective_interval = cfg_interval

    if effective_sample_path and effective_sample_path.exists():
        sample_cfg = {**sample_cfg, "sample_interval": effective_interval}
        from .processor import insert_sample_markers
        tasks = insert_sample_markers(
            tasks, sample_cfg, effective_sample_path,
            sku_boundary=not no_sku_boundary,
        )
        mode_parts = []
        if not no_sku_boundary:
            mode_parts.append("SKU首尾")
        if effective_interval > 0:
            mode_parts.append(f"每{effective_interval}张")
        console.print(f"[cyan]→[/cyan] 抽检样图: {effective_sample_path.name}（{' + '.join(mode_parts)}）")
    else:
        if effective_sample_path and not effective_sample_path.exists():
            console.print(f"[yellow]![/yellow] 抽检样图不存在: {effective_sample_path}")

    console.print(f"[cyan]→[/cyan] 开始处理 [bold]{len(tasks)}[/bold] 张图片")

    stats: BatchStats = BatchStats()
    manifest = Manifest()

    if dry_run:
        from .processor import process_single_image, _extract_sku_from_path
        for t in tasks:
            result = process_single_image(t, preset_cfg, global_cfg.get("keep_original_structure", True), overwrite, True)
            _update_stats(stats, result)
            is_sample = t.is_sample
            sample_tag = ""
            if is_sample:
                pos_str = t.sample_position.replace("_", " ")
                sample_tag = f"[magenta][抽检 {pos_str}][/magenta] "

            if result.dry_run_details:
                for d in result.dry_run_details:
                    sku_dir_str = ""
                    if t.sku_group_dir:
                        try:
                            rel_sku = t.sku_group_dir.relative_to(output_dir)
                            sku_dir_str = f" → [blue]{rel_sku}[/blue]/"
                        except ValueError:
                            sku_dir_str = f" → [blue]{t.sku_group_dir}[/blue]/"
                    console.print(
                        f"  {sample_tag}{result.input_path.name}{sku_dir_str}"
                        f"{d['filename']}  [{d['width']}×{d['height']}] "
                        f"{d['format']} q={d['quality']}"
                    )
        _print_stats(stats, output_dir, global_cfg.get("error_log", "errors.log"))
        return

    results_list: List[ProcessResult] = []

    def _collect_result(result: ProcessResult, done: int, total: int) -> None:
        results_list.append(result)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("处理中...", total=len(tasks))

        def _prog_cb(result: ProcessResult, done: int, total: int) -> None:
            _collect_result(result, done, total)
            name = result.input_path.name
            dur = f"{result.duration_ms}ms"
            if result.skipped:
                status = f"[yellow]跳过 {name}[/yellow] ({dur})"
            elif result.success:
                status = f"[green]✓ {name}[/green] ({dur})"
            else:
                status = f"[red]✗ {name}[/red] ({result.error.category.value if result.error else 'err'})"
            progress.update(task_id, completed=done, description=status)

        stats = batch_process(
            tasks=tasks,
            preset=preset_cfg,
            workers=workers,
            keep_structure=global_cfg.get("keep_original_structure", True) and not sku_group,
            overwrite=overwrite,
            dry_run=False,
            progress_callback=_prog_cb,
        )

    out_fmt = preset_cfg.get("output", {}).get("format", "webp")
    out_quality = preset_cfg.get("output", {}).get("quality", 75)
    final_sizes = preset_cfg.get("sizes", [])

    for result in results_list:
        t = tasks[result.task_index] if 0 <= result.task_index < len(tasks) else None
        if t is None:
            continue
        sku_key = _compute_sku_key(t)
        sku_dir = str(t.sku_group_dir) if t.sku_group_dir else ""
        sizes_info = []
        if result.dry_run_details:
            sizes_info = result.dry_run_details
        elif result.success:
            for sz in (final_sizes + (t.extra_sizes or [])):
                sizes_info.append({"name": sz.get("name", ""), "width": sz["width"], "height": sz["height"], "suffix": sz.get("suffix", "")})

        manifest.add_entry(
            input_path=t.input_path,
            output_paths=result.output_paths,
            sizes=sizes_info,
            fmt=out_fmt,
            quality=out_quality,
            duration_ms=result.duration_ms,
            is_sample=t.is_sample,
            sample_position=t.sample_position,
            success=result.success,
            error_category=result.error.category.value if result.error else "",
            error_message=result.error.message if result.error else "",
            sku_key=sku_key,
            sku_dir=sku_dir,
        )

    manifest.finalize(preset, input_dir, output_dir, stats.duration_ms)
    manifest_path = manifest.save(output_dir)
    console.print(f"[cyan]→[/cyan] 处理清单: {manifest_path}")

    _print_stats(stats, output_dir, global_cfg.get("error_log", "errors.log"))


def _print_stats(stats: BatchStats, output_dir: Path, error_log_name: str) -> None:
    console.print()
    table = Table(title="处理结果汇总", show_header=False, header_style="cyan", box=None)
    table.add_column(style="bold", width=15)
    table.add_column()
    table.add_row("总数", str(stats.total))
    table.add_row("成功", f"[green]{stats.success}[/green]")
    table.add_row("失败", f"[red]{stats.failed}[/red]")
    if stats.skipped:
        table.add_row("跳过", f"[yellow]{stats.skipped}[/yellow]")
    table.add_row("总耗时", f"{stats.duration_ms / 1000:.1f}s")
    if stats.total > 0:
        table.add_row("平均速度", f"{stats.duration_ms / max(stats.total, 1):.0f}ms/张")
    console.print(table)

    if stats.errors:
        err_table = Table(title="错误分类统计", header_style="red", show_lines=False)
        err_table.add_column("分类", style="bold")
        err_table.add_column("数量", justify="right")
        for cat, count in sorted(stats.error_counts.items(), key=lambda x: -x[1]):
            err_table.add_row(cat.value, str(count))
        console.print(err_table)

        log_path = output_dir / error_log_name
        write_errors_log(stats.errors, log_path)
        console.print(f"[red]✗[/red] 详细错误日志: {log_path}")

    console.print(f"\n[cyan]→[/cyan] 输出目录: {output_dir}")


def _hash_scan_impl(
    directory: Path,
    threshold: int = 8,
    action: str = "report",
    output: Optional[Path] = None,
    recursive: bool = True,
    yes: bool = False,
) -> None:
    hasher = ImageHasher(threshold=threshold)
    console.print(f"[cyan]→[/cyan] 扫描目录: {directory}")
    console.print(f"[cyan]→[/cyan] 汉明距离阈值: {threshold}")
    console.print(f"[cyan]→[/cyan] 操作: {action}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("计算哈希...", total=1)

        def _cb(idx: int, total: int, path: Path) -> None:
            progress.update(task_id, total=total, completed=idx, description=f"哈希: {path.name[:40]}")

        hash_infos = hasher.scan_directory(directory, recursive=recursive, progress_callback=_cb)

    console.print(f"[cyan]→[/cyan] 计算完成: {len(hash_infos)} 张")

    if len(hash_infos) < 2:
        console.print("[yellow]![/yellow] 图片太少，无法比较")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("比对重复...", total=len(hash_infos))

        def _cb2(idx: int, total: int) -> None:
            progress.update(task_id, completed=idx)

        pairs = hasher.find_duplicates(hash_infos, progress_callback=_cb2)

    if not pairs:
        console.print("[green]✓[/green] 未发现重复图片")
        return

    groups = hasher.group_duplicates(pairs)
    console.print(f"[yellow]⚠[/yellow] 发现 [bold]{len(groups)}[/bold] 组重复，涉及 [bold]{sum(len(g.images) for g in groups)}[/bold] 张")

    table = Table(title="重复分组详情", show_lines=False, header_style="yellow")
    table.add_column("#", style="bold", justify="right")
    table.add_column("推荐保留", style="green")
    table.add_column("分辨率")
    table.add_column("评分", justify="right")
    table.add_column("重复数", justify="right")
    for i, g in enumerate(groups, 1):
        keep = g.images[g.keep_index]
        table.add_row(
            str(i),
            keep.path.name,
            f"{keep.resolution[0]}x{keep.resolution[1]}",
            f"{keep.score:.3f}",
            str(len(g.images)),
        )
    console.print(table)

    if action == "report":
        return

    out_dir = output or directory
    use_symlinks = action == "link"

    should_proceed = yes or Confirm.ask(
        f"确认将重复文件{'软链接' if use_symlinks else '复制'}到 {out_dir / 'dupes'} ?", default=True
    )
    if should_proceed:
        stats2 = hasher.organize_duplicates(groups, out_dir, use_symlinks=use_symlinks)
        console.print(
            f"[green]✓[/green] 完成: {stats2['groups']} 组, "
            f"软链接 {stats2['linked']} 个, 复制 {stats2['moved']} 个"
        )


app.add_typer(hash_app, name="hash-scan")


@app.command("report")
def cmd_report(
    output_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="输出目录（含manifest.json）"),
    csv_path: Optional[Path] = typer.Option(None, "--csv", "-c", help="导出CSV文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示每张文件详情"),
) -> None:
    """读取 manifest 和 errors.log，汇总处理结果，可导出 CSV"""
    _banner()

    manifest = Manifest.load(output_dir)
    if manifest is None:
        err_console.print(f"[red]✗[/red] 未找到 {output_dir / Manifest.FILENAME}，请先运行 picflow process")
        raise typer.Exit(1)

    summary = manifest.get_summary()

    console.print(f"[cyan]→[/cyan] 输出目录: {output_dir}")
    console.print(f"[cyan]→[/cyan] 处理时间: {manifest.meta.created_at}")
    console.print(f"[cyan]→[/cyan] 使用预设: {manifest.meta.preset}")

    main_table = Table(title="处理结果汇总", show_header=False, header_style="cyan", box=None)
    main_table.add_column(style="bold", width=18)
    main_table.add_column()
    main_table.add_row("输入图片", str(summary["total_input"]))
    main_table.add_row("成功", f"[green]{summary['total_success']}[/green]")
    main_table.add_row("失败", f"[red]{summary['total_failed']}[/red]")
    main_table.add_row("抽检图", f"[magenta]{summary['total_samples']}[/magenta]")
    main_table.add_row("输出文件数", str(summary["total_output_files"]))
    main_table.add_row("总耗时", f"{summary['total_duration_ms'] / 1000:.1f}s")
    console.print(main_table)

    if summary["sku_counts"]:
        sku_table = Table(title="SKU分组统计", header_style="blue", show_lines=False)
        sku_table.add_column("SKU", style="bold")
        sku_table.add_column("图片数", justify="right")
        for sku_key, count in sorted(summary["sku_counts"].items(), key=lambda x: -x[1]):
            sku_table.add_row(sku_key, str(count))
        console.print(sku_table)

    if summary["size_counts"]:
        size_table = Table(title="输出尺寸统计", header_style="green", show_lines=False)
        size_table.add_column("尺寸", style="bold")
        size_table.add_column("数量", justify="right")
        for size_key, count in sorted(summary["size_counts"].items(), key=lambda x: -x[1]):
            size_table.add_row(size_key, str(count))
        console.print(size_table)

    if summary["error_counts"]:
        err_table = Table(title="错误分类统计", header_style="red", show_lines=False)
        err_table.add_column("分类", style="bold")
        err_table.add_column("数量", justify="right")
        for cat, count in sorted(summary["error_counts"].items(), key=lambda x: -x[1]):
            err_table.add_row(cat, str(count))
        console.print(err_table)

    if summary["sample_positions"]:
        sample_table = Table(title="抽检图插入位置", header_style="magenta", show_lines=False)
        sample_table.add_column("#", justify="right")
        sample_table.add_column("位置")
        for i, pos in enumerate(summary["sample_positions"], 1):
            sample_table.add_row(str(i), pos)
        console.print(sample_table)

    if verbose:
        detail_table = Table(title="逐文件详情", show_lines=False, header_style="dim")
        detail_table.add_column("输入", style="bold", max_width=40)
        detail_table.add_column("SKU目录", max_width=25)
        detail_table.add_column("状态", max_width=8)
        detail_table.add_column("耗时", justify="right", max_width=8)
        detail_table.add_column("抽检", max_width=10)
        for e in manifest.entries:
            input_name = Path(e.input_path).name
            sku_short = ""
            if e.sku_dir:
                try:
                    sku_short = str(Path(e.sku_dir).relative_to(manifest.meta.output_dir))
                except ValueError:
                    sku_short = e.sku_dir
            status = "[green]✓[/green]" if e.success else f"[red]✗ {e.error_category}[/red]"
            dur = f"{e.duration_ms}ms"
            sample_str = e.sample_position if e.is_sample else ""
            detail_table.add_row(input_name, sku_short, status, dur, sample_str)
        console.print(detail_table)

    if csv_path:
        csv_path = manifest.export_csv(csv_path)
        console.print(f"[green]✓[/green] CSV已导出: {csv_path}")

    error_log = output_dir / "errors.log"
    if error_log.exists():
        console.print(f"[dim]详细错误日志: {error_log}[/dim]")


exif_app = typer.Typer(help="EXIF信息查看")
app.add_typer(exif_app, name="exif")


@exif_app.command("show")
def exif_show(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="保存为JSON"),
) -> None:
    """查看图片EXIF信息"""
    tags = read_exif_tags(image)
    if not tags:
        console.print("[yellow]![/yellow] 未找到EXIF信息")
        return

    table = Table(title=f"EXIF: {image.name}", show_lines=False, header_style="magenta")
    table.add_column("标签", style="bold")
    table.add_column("值")
    for k, v in tags.items():
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        table.add_row(k, val_str)
    console.print(table)

    if save:
        with open(save, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓[/green] 已保存到: {save}")


def main() -> None:
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        app()
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        err_console.print("\n[yellow]![/yellow] 操作已取消")
        sys.exit(130)
    except Exception as e:
        err_console.print(f"\n[red]✗[/red] 未处理的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
