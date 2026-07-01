"""
Training log structure discovery and metric extraction from raw directories.

Scans experiment directories, detects training framework artifacts,
and extracts structured metric series with framework-specific knowledge.
Handles PyTorch Lightning, HuggingFace Trainer, and JAX orbax checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .formatter import SQLFormatter, FormatResult, MetricKind, MetricPoint, classify_metric

logger = logging.getLogger("pulse.checker")


class Framework(Enum):
    PYTORCH = "pytorch"
    PYTORCH_LIGHTNING = "pytorch_lightning"
    HUGGINGFACE = "huggingface"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    DEEPSPEED = "deepspeed"
    UNKNOWN = "unknown"


@dataclass
class Experiment:
    """A discovered training experiment with all associated snapshots."""

    experiment_id: str
    root_path: str
    framework: Framework = Framework.UNKNOWN
    snapshots: List[FormatResult] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshots)

    @property
    def total_steps(self) -> int:
        return max((s.total_steps for s in self.snapshots), default=0)

    def aggregate_metric(self, name: str) -> List[MetricPoint]:
        all_points: List[MetricPoint] = []
        for snap in self.snapshots:
            all_points.extend(snap.metrics.get(name, []))
        all_points.sort(key=lambda p: p.step)
        return all_points


class FormatMetric:
    """Extracts structured metric series from FormatResult objects.

    Provides smoothing, interpolation, plateau detection, and
    convergence analysis for individual metric series.
    """

    @staticmethod
    def smooth(points: List[MetricPoint], window: int = 10) -> List[float]:
        if not points:
            return []
        sorted_pts = sorted(points, key=lambda p: p.step)
        values = [p.value for p in sorted_pts]
        if window <= 1 or len(values) <= window:
            return values
        result: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            win = values[start:i + 1]
            result.append(sum(win) / len(win))
        return result

    @staticmethod
    def downsample(
        points: List[MetricPoint], target_steps: int = 100
    ) -> List[MetricPoint]:
        if len(points) <= target_steps:
            return list(points)
        sorted_pts = sorted(points, key=lambda p: p.step)
        stride = len(sorted_pts) / target_steps
        result: List[MetricPoint] = []
        for i in range(target_steps):
            idx = int(i * stride)
            result.append(sorted_pts[idx])
        return result

    @staticmethod
    def find_plateau(
        points: List[MetricPoint],
        patience: int = 10,
        min_delta: float = 1e-4,
        minimize: bool = True,
    ) -> Optional[int]:
        """Find the step where improvement stops (plateau detection)."""
        if len(points) < patience:
            return None
        sorted_pts = sorted(points, key=lambda p: p.step)
        best_step = sorted_pts[0].step
        best_val = sorted_pts[0].value
        steps_since_improvement = 0
        for pt in sorted_pts[1:]:
            improved = (pt.value < best_val - min_delta) if minimize else (pt.value > best_val + min_delta)
            if improved:
                best_val = pt.value
                best_step = pt.step
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            if steps_since_improvement >= patience:
                return best_step
        return None

    @staticmethod
    def convergence_score(points: List[MetricPoint]) -> float:
        """Estimate convergence quality from the trajectory shape.

        Returns a score 0-1, higher = better convergence.
        """
        if len(points) < 10:
            return 0.0
        sorted_pts = sorted(points, key=lambda p: p.step)
        values = [p.value for p in sorted_pts]
        total = values[-1]
        best = min(values)
        if best <= 1e-12 or total <= 1e-12:
            return 1.0 if best <= 1e-12 else 0.0
        first_third = sum(values[:len(values)//3]) / max(1, len(values)//3)
        last_third = sum(values[-len(values)//3:]) / max(1, len(values)//3)
        relative_improvement = (first_third - last_third) / max(first_third, 1e-9)
        noise = sum(
            abs(values[i] - values[i - 1])
            for i in range(1, min(50, len(values)))
        ) / min(50, len(values))
        plateau_score = 1.0 - min(noise / max(abs(values[-1]), 1e-9), 1.0)
        return max(0.0, min(1.0, (relative_improvement * 0.7 + plateau_score * 0.3)))


class StyleChecker:
    """Discovers training experiment directories and extracts metrics."""

    FRAMEWORK_MARKERS: Dict[Framework, List[str]] = {
        Framework.PYTORCH: ["*.pt", "*.pth", "pytorch_model.bin"],
        Framework.PYTORCH_LIGHTNING: ["lightning_logs", "*.ckpt"],
        Framework.HUGGINGFACE: [
            "trainer_state.json", "training_args.bin",
            "adapter_config.json", "pytorch_model.bin",
        ],
        Framework.TENSORFLOW: [
            "events.out.tfevents.*", "checkpoint", "*.ckpt.index",
            "saved_model.pb",
        ],
        Framework.JAX: ["orbax-checkpoint*", "flax_model", "checkpoint_*"],
        Framework.DEEPSPEED: [
            "deepspeed_*", "bf16_zero_pp_*", "zero_to_fp32.py",
        ],
    }

    CONFIG_FILES = [
        "config.json", "hparams.yaml", "hparams.json",
        "trainer_config.yaml", "training_config.json",
        "hydra_config.yaml", ".hydra/config.yaml",
        "args.json", "opt.yaml",
    ]

    def __init__(self, formatter: Optional[SQLFormatter] = None):
        self.formatter = formatter or SQLFormatter()

    def discover(self, root: Path) -> Iterator[Experiment]:
        """Walk a directory tree and yield discovered experiments."""
        logger.info("Discovering experiments in %s", root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            framework = self._detect_framework(Path(dirpath), filenames)
            if framework == Framework.UNKNOWN:
                continue
            experiment = Experiment(
                experiment_id=self._make_id(dirpath),
                root_path=str(dirpath),
                framework=framework,
            )
            experiment.config = self._load_config(Path(dirpath))
            experiment.checkpoints = self._find_checkpoints(Path(dirpath))
            for log_file in self._find_log_files(Path(dirpath), framework):
                snapshot = self.formatter.parse(log_file)
                if snapshot and snapshot.metrics:
                    experiment.snapshots.append(snapshot)
            if experiment.snapshots:
                yield experiment

    def scan_directory(self, root: Path) -> List[Experiment]:
        return list(self.discover(root))

    def _detect_framework(self, dirpath: Path, filenames: List[str]) -> Framework:
        for fw, markers in self.FRAMEWORK_MARKERS.items():
            for marker in markers:
                if marker.startswith("*"):
                    if list(dirpath.glob(marker)):
                        return fw
                elif marker in filenames:
                    return fw
        # Check for training log patterns in text files
        for fname in filenames:
            if fname.endswith((".log", ".txt", ".out")):
                try:
                    sample = (dirpath / fname).read_text()[:4096]
                    if re.search(r"(?:Epoch|Step|Iter)\s*[\[=:]\s*\d+", sample):
                        if "loss" in sample.lower():
                            if re.search(r"pl\.|lightning|Trainer", sample):
                                return Framework.PYTORCH_LIGHTNING
                            return Framework.PYTORCH
                except (OSError, UnicodeDecodeError):
                    pass
        return Framework.UNKNOWN

    def _load_config(self, dirpath: Path) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for cfg_file in self.CONFIG_FILES:
            cfg_path = dirpath / cfg_file
            if not cfg_path.is_file():
                continue
            try:
                text = cfg_path.read_text(encoding="utf-8")
                if cfg_file.endswith((".yaml", ".yml")):
                    try:
                        import yaml
                        data = yaml.safe_load(text)
                    except ImportError:
                        continue
                else:
                    data = json.loads(text)
                if isinstance(data, dict):
                    config.update(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("Failed to parse config %s: %s", cfg_path, e)
        return config

    def _find_checkpoints(self, dirpath: Path) -> List[str]:
        ckpt_patterns = ["*.pt", "*.pth", "*.ckpt", "*.bin", "checkpoint-*"]
        results: List[str] = []
        for pat in ckpt_patterns:
            for p in dirpath.glob(pat):
                if p.is_file():
                    results.append(str(p))
        return sorted(results)[:50]

    def _find_log_files(self, dirpath: Path, framework: Framework) -> List[Path]:
        log_files: List[Path] = []
        for ext in ("*.log", "*.txt", "*.out", "*.json", "*.jsonl", "*.csv"):
            for p in dirpath.glob(ext):
                if p.is_file() and p.stat().st_size < 50 * 1024 * 1024:
                    log_files.append(p)
        # TensorBoard events
        for p in dirpath.glob("events.out.tfevents.*"):
            log_files.append(p)
        return sorted(log_files)[:100]

    @staticmethod
    def _make_id(dirpath: str) -> str:
        import hashlib
        return hashlib.md5(dirpath.encode()).hexdigest()[:12]
