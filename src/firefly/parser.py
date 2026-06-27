"""
Structured log parser for training frameworks (PyTorch, TensorFlow, JAX).

Detects log format automatically, extracts step-level metrics, timestamps,
loss curves, learning rates, and GPU utilization from heterogeneous log files.
Supports TensorBoard event files, W&B logs, and plain-text training outputs.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger("firefly.parser")


class LogFormat(Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    PLAINTEXT = "plaintext"
    JSONL = "jsonl"
    CSV = "csv"
    UNKNOWN = "unknown"


class MetricKind(Enum):
    LOSS = auto()
    LEARNING_RATE = auto()
    GRADIENT_NORM = auto()
    ACCURACY = auto()
    PERPLEXITY = auto()
    THROUGHPUT = auto()
    GPU_UTIL = auto()
    GPU_MEMORY = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class MetricPoint:
    """A single metric observation at a specific training step."""

    step: int
    value: float
    metric_name: str
    kind: MetricKind = MetricKind.CUSTOM
    wall_time: float = 0.0
    epoch: int = 0
    rank: int = 0  # For distributed training
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LogSnapshot:
    """A parsed snapshot of training state from a log file or directory."""

    source_path: str
    format: LogFormat
    captured_at: datetime = field(default_factory=datetime.now)
    total_steps: int = 0
    epochs: int = 0
    metrics: Dict[str, List[MetricPoint]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    raw_lines: int = 0
    parse_errors: int = 0

    @property
    def metric_names(self) -> List[str]:
        return sorted(self.metrics.keys())

    def range(self, name: str, start_step: int = 0,
              end_step: Optional[int] = None) -> List[MetricPoint]:
        points = self.metrics.get(name, [])
        end = end_step if end_step is not None else float("inf")
        return [p for p in points if start_step <= p.step <= end]

    def final_value(self, name: str) -> Optional[float]:
        points = self.metrics.get(name, [])
        if not points:
            return None
        return max(points, key=lambda p: p.step).value

    def best_value(self, name: str, minimize: bool = True) -> Optional[float]:
        points = self.metrics.get(name, [])
        if not points:
            return None
        return (min if minimize else max)(p.value for p in points)

    def smoothing(self, name: str, window: int = 10) -> List[float]:
        points = sorted(self.metrics.get(name, []), key=lambda p: p.step)
        if not points:
            return []
        values = [p.value for p in points]
        smoothed: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start:i + 1]
            smoothed.append(sum(window_vals) / len(window_vals))
        return smoothed


# ——— Metric classification ——————————————————————————————————————————


_METRIC_CLASSIFIERS: Dict[MetricKind, List[str]] = {
    MetricKind.LOSS: [
        r"loss", r"cost", r"objective", r"error", r"cross.entropy",
        r"nll", r"mse", r"mae", r"bce", r"kl.div",
    ],
    MetricKind.LEARNING_RATE: [
        r"(?:learning.?rate|lr|eta)(?!.*(?:loss|decay))",
    ],
    MetricKind.GRADIENT_NORM: [
        r"grad(?:ient)?.?norm", r"grad.?l2", r"gnorm",
    ],
    MetricKind.ACCURACY: [
        r"acc(?:uracy)?", r"top.[1-5]", r"f1", r"precision", r"recall",
        r"bleu", r"rouge", r"perplexity",
    ],
    MetricKind.PERPLEXITY: [r"perplexity", r"ppl"],
    MetricKind.THROUGHPUT: [
        r"(?:throughput|samples.?/s|tokens.?/s|steps.?/s|it.?/s)",
        r"(?:img|image)s?/s", r"batch(?:es)?/s",
    ],
    MetricKind.GPU_UTIL: [r"gpu.?util", r"utilization"],
    MetricKind.GPU_MEMORY: [r"gpu.?mem", r"(?:allocated|reserved|memory).*?(?:mb|gb|mib|gib)"],
}


def classify_metric(name: str) -> MetricKind:
    name_lower = name.lower().replace("_", " ").replace("-", " ")
    for kind, patterns in _METRIC_CLASSIFIERS.items():
        for pat in patterns:
            if re.search(pat, name_lower):
                return kind
    return MetricKind.CUSTOM


# ——— Parser implementations ——————————————————————————————————————————


class LogParser:
    """Multi-format log parser with automatic format detection.

    Accepts individual log files or directories. Detects format by
    extension and content sampling, then dispatches to specialized
    parsers for each supported format.
    """

    def __init__(self):
        self._line_parsers: Dict[LogFormat, Callable] = {
            LogFormat.PLAINTEXT: self._parse_plaintext,
            LogFormat.JSONL: self._parse_jsonl,
            LogFormat.CSV: self._parse_csv,
        }

    def detect_format(self, path: Path) -> LogFormat:
        """Detect the log format of a file or directory."""
        if path.is_dir():
            if (path / "events.out.tfevents").exists() or list(path.glob("events.out.tfevents*")):
                return LogFormat.TENSORBOARD
            if (path / "wandb").exists():
                return LogFormat.WANDB
            return LogFormat.UNKNOWN
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".json"):
            return LogFormat.JSONL
        if suffix == ".csv":
            return LogFormat.CSV
        if suffix in (".gz", ".tgz") and ".tfevents" in path.name:
            return LogFormat.TENSORBOARD
        # Sample content to detect
        try:
            sample = path.read_text(encoding="utf-8")[:4096]
            if sample.startswith("[") or sample.startswith("{"):
                try:
                    json.loads(sample.split("\n")[0])
                    return LogFormat.JSONL
                except json.JSONDecodeError:
                    pass
            if "," in sample and "\n" in sample:
                if re.match(r"^[\w\s,._-]+$", sample.split("\n")[0]):
                    return LogFormat.CSV
        except (OSError, UnicodeDecodeError):
            pass
        return LogFormat.PLAINTEXT

    def parse(self, path: Path) -> LogSnapshot:
        """Parse a log file or directory into a structured snapshot."""
        fmt = self.detect_format(path)
        snapshot = LogSnapshot(
            source_path=str(path),
            format=fmt,
        )
        if fmt in (LogFormat.TENSORBOARD, LogFormat.WANDB):
            logger.info("Binary format detected: %s — use parse_events()", fmt)
            return snapshot
        parser = self._line_parsers.get(fmt)
        if parser is None:
            logger.warning("No parser for format %s", fmt)
            return snapshot
        try:
            parser(path, snapshot)
        except Exception as e:
            logger.error("Parse error for %s: %s", path, e)
            snapshot.parse_errors += 1
        return snapshot

    def parse_directory(self, path: Path) -> List[LogSnapshot]:
        snapshots: List[LogSnapshot] = []
        for fpath in sorted(path.rglob("*")):
            if fpath.is_file() and not fpath.name.startswith("."):
                try:
                    snapshots.append(self.parse(fpath))
                except Exception:
                    logger.debug("Skipping %s", fpath, exc_info=True)
        return snapshots

    def _parse_plaintext(self, path: Path, snapshot: LogSnapshot) -> None:
        """Parse a plain-text training log with regex-based metric extraction."""
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        snapshot.raw_lines = len(lines)

        # Common training log patterns
        patterns: List[Tuple[re.Pattern, Dict[str, int]]] = [
            # PyTorch-style: "Epoch [1/50], Step [100/500], Loss: 2.3456, LR: 0.001"
            (re.compile(
                r"(?:Epoch|E)\s*\[?(\d+).*?(?:Step|Iter|I)\s*\[?(\d+).*?"
                r"(?:Loss|L|cost)[=:\s]*([\d.eE+-]+).*?"
                r"(?:LR|lr|Learning Rate)[=:\s]*([\d.eE+-]+)",
                re.IGNORECASE
            ), {"epoch": 1, "step": 2, "loss": 3, "lr": 4}),
            # Generic: "step 100: loss=2.345, acc=0.876"
            (re.compile(
                r"(?:step|iter(?:ation)?)[=:\s]*(\d+).*?"
                r"(\w[\w._-]*)[=:\s]*([\d.eE+-]+)",
                re.IGNORECASE
            ), {"step": 1, "metric_name": 2, "metric_value": 3}),
            # TensorFlow: "Step 100: loss = 2.345 (0.123 sec)"
            (re.compile(
                r"[Ss]tep\s+(\d+).*?(\w[\w._-]*)\s*=\s*([\d.eE+-]+)",
            ), {"step": 1, "metric_name": 2, "metric_value": 3}),
        ]

        for line in lines:
            line = line.strip()
            if not line or line[0] in "#;/":
                continue
            for pat, groups in patterns:
                m = pat.search(line)
                if m:
                    try:
                        step_str = m.group(groups.get("step", 0) or 0)
                        step = int(step_str) if step_str else 0
                    except (ValueError, IndexError):
                        continue
                    if "loss" in groups:
                        try:
                            loss = float(m.group(groups["loss"]))
                            self._add_metric(snapshot, "loss", loss, step, MetricKind.LOSS)
                        except (ValueError, IndexError):
                            pass
                    if "lr" in groups:
                        try:
                            lr = float(m.group(groups["lr"]))
                            self._add_metric(snapshot, "learning_rate", lr, step, MetricKind.LEARNING_RATE)
                        except (ValueError, IndexError):
                            pass
                    if "metric_name" in groups and "metric_value" in groups:
                        try:
                            name = m.group(groups["metric_name"])
                            val = float(m.group(groups["metric_value"]))
                            kind = classify_metric(name)
                            self._add_metric(snapshot, name, val, step, kind)
                        except (ValueError, IndexError):
                            pass
        snapshot.total_steps = max(
            (max((p.step for p in pts), default=0) for pts in snapshot.metrics.values()),
            default=0,
        )

    def _parse_jsonl(self, path: Path, snapshot: LogSnapshot) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                snapshot.parse_errors += 1
                continue
            step = record.get("step") or record.get("iteration") or record.get("iter") or 0
            epoch = record.get("epoch") or 0
            for key, val in record.items():
                if key in ("step", "iteration", "iter", "epoch", "timestamp", "wall_time"):
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    kind = classify_metric(key)
                    mp = MetricPoint(
                        step=int(step), value=float(val), metric_name=key,
                        kind=kind, epoch=int(epoch),
                    )
                    snapshot.metrics.setdefault(key, []).append(mp)
        snapshot.total_steps = max(
            (max((p.step for p in pts), default=0) for pts in snapshot.metrics.values()),
            default=0,
        )

    def _parse_csv(self, path: Path, snapshot: LogSnapshot) -> None:
        import csv
        import io
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        step_col = None
        for possible in ("step", "Step", "iteration", "iter", "epoch"):
            if possible in (reader.fieldnames or []):
                step_col = possible
                break
        for row in reader:
            step = int(row.get(step_col or "", 0)) if step_col else snapshot.total_steps
            for key, val in row.items():
                if key == step_col:
                    continue
                try:
                    float_val = float(val)
                except (ValueError, TypeError):
                    continue
                kind = classify_metric(key)
                self._add_metric(snapshot, key, float_val, step, kind)
            snapshot.total_steps = max(snapshot.total_steps, step)

    @staticmethod
    def _add_metric(
        snapshot: LogSnapshot, name: str, value: float,
        step: int, kind: MetricKind = MetricKind.CUSTOM,
    ) -> None:
        mp = MetricPoint(step=step, value=value, metric_name=name, kind=kind)
        snapshot.metrics.setdefault(name, []).append(mp)
