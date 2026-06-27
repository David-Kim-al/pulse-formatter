"""
Snapshot comparison engine for training experiments.

Compares training runs across experiments, generates diff summaries,
computes statistical significance of metric differences, and identifies
regressions or improvements between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .parser import LogSnapshot, MetricKind, MetricPoint
from .discoverer import Experiment, MetricExtractor


@dataclass
class MetricDiff:
    """Difference between two metric series."""

    metric_name: str
    values_a: List[float]
    values_b: List[float]
    mean_a: float
    mean_b: float
    delta: float
    delta_pct: float
    significant: bool
    p_value: float
    direction: str  # "improved", "regressed", "similar"

    @property
    def summary(self) -> str:
        arrow = "↓" if self.direction == "improved" else ("↑" if self.direction == "regressed" else "→")
        sig = "*" if self.significant else ""
        return (
            f"{self.metric_name}: {self.mean_a:.4f} → {self.mean_b:.4f} "
            f"({self.delta_pct:+.1f}%) {arrow}{sig}"
        )


@dataclass
class ExperimentDiff:
    """Full comparison between two experiments or snapshots."""

    name_a: str
    name_b: str
    metric_diffs: List[MetricDiff]
    step_diff: int
    config_diffs: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    @property
    def improved_metrics(self) -> List[MetricDiff]:
        return [d for d in self.metric_diffs if d.direction == "improved"]

    @property
    def regressed_metrics(self) -> List[MetricDiff]:
        return [d for d in self.metric_diffs if d.direction == "regressed"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.name_a,
            "b": self.name_b,
            "step_diff": self.step_diff,
            "diffs": [
                {
                    "metric": d.metric_name,
                    "mean_a": d.mean_a,
                    "mean_b": d.mean_b,
                    "delta_pct": round(d.delta_pct, 2),
                    "significant": d.significant,
                    "direction": d.direction,
                }
                for d in self.metric_diffs
            ],
        }


class SnapshotComparator:
    """Statistical comparison of training log snapshots.

    Supports paired and unpaired comparison, Welch's t-test,
    and effect size calculation (Cohen's d).
    """

    MIN_SAMPLES_FOR_TEST = 5
    SIGNIFICANCE_LEVEL = 0.05
    SMALL_EFFECT = 0.2
    MEDIUM_EFFECT = 0.5
    LARGE_EFFECT = 0.8

    def __init__(self, significance: float = SIGNIFICANCE_LEVEL):
        self.significance = significance

    def compare_snapshots(
        self, a: LogSnapshot, b: LogSnapshot
    ) -> ExperimentDiff:
        """Compare two log snapshots metric by metric."""
        common_metrics = set(a.metric_names) & set(b.metric_names)
        diffs: List[MetricDiff] = []
        for name in sorted(common_metrics):
            points_a = a.metrics.get(name, [])
            points_b = b.metrics.get(name, [])
            if not points_a or not points_b:
                continue
            vals_a = [p.value for p in points_a]
            vals_b = [p.value for p in points_b]
            diff = self._compute_diff(name, vals_a, vals_b)
            diffs.append(diff)
        return ExperimentDiff(
            name_a=a.source_path,
            name_b=b.source_path,
            metric_diffs=diffs,
            step_diff=b.total_steps - a.total_steps,
        )

    def compare_experiments(
        self, exp_a: Experiment, exp_b: Experiment
    ) -> ExperimentDiff:
        """Compare two experiments by aggregating all snapshots."""
        all_metrics: Dict[str, List[List[float]]] = {}
        for exp, key in [(exp_a, "a"), (exp_b, "b")]:
            for snap in exp.snapshots:
                for name in snap.metric_names:
                    vals = [p.value for p in snap.metrics.get(name, [])]
                    if vals:
                        all_metrics.setdefault(name, [[], []])[0 if key == "a" else 1].extend(vals)
        diffs: List[MetricDiff] = []
        for name, (vals_a, vals_b) in all_metrics.items():
            if vals_a and vals_b:
                diffs.append(self._compute_diff(name, vals_a, vals_b))
        return ExperimentDiff(
            name_a=exp_a.experiment_id,
            name_b=exp_b.experiment_id,
            metric_diffs=diffs,
            step_diff=exp_b.total_steps - exp_a.total_steps,
        )

    def find_regressions(
        self, diffs: List[MetricDiff], metric_kinds: Optional[List[str]] = None
    ) -> List[MetricDiff]:
        """Find metrics that regressed significantly.

        Loss metrics: higher = regression. Accuracy metrics: lower = regression.
        """
        loss_like = {"loss", "cost", "error", "nll", "mse", "bce"}
        regressed: List[MetricDiff] = []
        for d in diffs:
            is_loss = any(kw in d.metric_name.lower() for kw in loss_like)
            if is_loss and d.direction == "regressed":
                regressed.append(d)
            elif not is_loss and d.direction == "regressed":
                regressed.append(d)
        return regressed

    def _compute_diff(
        self, name: str, vals_a: List[float], vals_b: List[float]
    ) -> MetricDiff:
        mean_a = sum(vals_a) / len(vals_a)
        mean_b = sum(vals_b) / len(vals_b)
        delta = mean_b - mean_a
        delta_pct = (delta / abs(mean_a)) * 100 if abs(mean_a) > 1e-9 else 0
        significant, p_val = self._welch_ttest(vals_a, vals_b)
        direction = self._assess_direction(name, delta, significant)
        return MetricDiff(
            metric_name=name,
            values_a=vals_a, values_b=vals_b,
            mean_a=mean_a, mean_b=mean_b,
            delta=delta, delta_pct=delta_pct,
            significant=significant, p_value=p_val,
            direction=direction,
        )

    @staticmethod
    def _welch_ttest(
        a: List[float], b: List[float]
    ) -> Tuple[bool, float]:
        """Welch's t-test for unequal variances."""
        n_a, n_b = len(a), len(b)
        if n_a < SnapshotComparator.MIN_SAMPLES_FOR_TEST or n_b < SnapshotComparator.MIN_SAMPLES_FOR_TEST:
            return False, 1.0
        mean_a = sum(a) / n_a
        mean_b = sum(b) / n_b
        var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1) if n_b > 1 else 0
        if var_a < 1e-12 and var_b < 1e-12:
            return False, 1.0
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-12:
            return False, 1.0
        t_stat = (mean_a - mean_b) / se
        df_num = (var_a / n_a + var_b / n_b) ** 2
        df_den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
        df = df_num / df_den if df_den > 1e-12 else 20
        p_val = SnapshotComparator._t_survival(abs(t_stat), df)
        return p_val < 0.05, p_val

    @staticmethod
    def _t_survival(t: float, df: float) -> float:
        """Approximate Student's t survival function."""
        x = df / (df + t * t)
        a = 0.5
        b = 0.5 * df
        xx = 1.0
        s = 1.0
        for k in range(1, 100):
            xx *= (b + k - 1) * x / k
            s += xx
            if abs(xx) < 1e-15:
                break
        result = x ** a * s / (a + 0.5 * df)
        if t > 0:
            result = 0.5 * result / (a + 0.5 * df) if df < 100 else result
        return min(result * 2, 1.0)  # two-tailed

    @staticmethod
    def _assess_direction(
        metric_name: str, delta: float, significant: bool
    ) -> str:
        if not significant:
            return "similar"
        name_lower = metric_name.lower()
        loss_like = any(
            kw in name_lower
            for kw in ("loss", "cost", "error", "nll", "mse", "bce", "perplexity")
        )
        if loss_like:
            return "improved" if delta < 0 else "regressed"
        else:
            return "improved" if delta > 0 else "regressed"
