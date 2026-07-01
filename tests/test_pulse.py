"""Tests for Pulse log formatter, checker, and comparator."""

import tempfile
from pathlib import Path

from pulse.formatter import (
    SQLFormatter, FormatResult, LogFormat, MetricPoint, MetricKind, classify_metric,
)
from pulse.checker import StyleChecker, FormatMetric, Experiment
from pulse.comparator import FormatComparator, MetricDiff


class TestSQLFormatter:
    def test_detect_jsonl(self):
        formatter = SQLFormatter()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write('{"step": 1, "loss": 2.5, "lr": 0.001}\n')
            f.write('{"step": 2, "loss": 2.3, "lr": 0.0009}\n')
            f.flush()
            snap = formatter.parse(Path(f.name))
            assert snap.format == LogFormat.JSONL
            assert "loss" in snap.metric_names
            Path(f.name).unlink()

    def test_parse_jsonl_metrics(self):
        formatter = SQLFormatter()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write('{"step": 10, "loss": 0.5, "accuracy": 0.9}\n')
            f.flush()
            snap = formatter.parse(Path(f.name))
            assert snap.metrics["loss"][0].value == 0.5
            assert snap.metrics["accuracy"][0].value == 0.9
            Path(f.name).unlink()

    def test_detect_plaintext(self):
        formatter = SQLFormatter()
        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
            f.write("Step 100: loss=1.234, acc=0.876\n")
            f.write("Step 200: loss=0.987, acc=0.912\n")
            f.flush()
            snap = formatter.parse(Path(f.name))
            assert snap.raw_lines >= 1
            Path(f.name).unlink()


class TestFormatMetric:
    def test_smooth(self):
        pts = [MetricPoint(step=i, value=float(i % 5), metric_name="test") for i in range(20)]
        smoothed = FormatMetric.smooth(pts, window=5)
        assert len(smoothed) == 20

    def test_downsample(self):
        pts = [MetricPoint(step=i, value=float(i), metric_name="test") for i in range(1000)]
        result = FormatMetric.downsample(pts, target_steps=50)
        assert len(result) <= 50

    def test_find_plateau_decreasing(self):
        pts = [
            MetricPoint(step=i, value=max(0.01, 1.0 / (i + 1)), metric_name="loss")
            for i in range(50)
        ]
        step = FormatMetric.find_plateau(pts, patience=10, minimize=True)
        assert step is not None

    def test_convergence_score(self):
        pts = [
            MetricPoint(step=i, value=1.0 / (i + 1) + 0.01, metric_name="loss")
            for i in range(100)
        ]
        score = FormatMetric.convergence_score(pts)
        assert 0 <= score <= 1


class TestClassifyMetric:
    def test_loss_metric(self):
        assert classify_metric("train_loss") == MetricKind.LOSS
        assert classify_metric("val_cross_entropy") == MetricKind.LOSS

    def test_lr_metric(self):
        assert classify_metric("learning_rate") == MetricKind.LEARNING_RATE
        assert classify_metric("lr") == MetricKind.LEARNING_RATE

    def test_accuracy_metric(self):
        assert classify_metric("accuracy") == MetricKind.ACCURACY
        assert classify_metric("top_1_acc") == MetricKind.ACCURACY

    def test_custom_metric(self):
        assert classify_metric("some_random_metric") == MetricKind.CUSTOM


class TestFormatComparator:
    def test_compare_snapshots(self):
        snap_a = FormatResult(source_path="a.log", format=LogFormat.JSONL)
        snap_a.metrics["loss"] = [
            MetricPoint(step=i, value=float(10 - i * 0.1), metric_name="loss")
            for i in range(10)
        ]
        snap_b = FormatResult(source_path="b.log", format=LogFormat.JSONL)
        snap_b.metrics["loss"] = [
            MetricPoint(step=i, value=float(10 - i * 0.08), metric_name="loss")
            for i in range(10)
        ]
        comparator = FormatComparator()
        diff = comparator.compare_snapshots(snap_a, snap_b)
        assert len(diff.metric_diffs) == 1
        assert "loss" in diff.metric_diffs[0].metric_name

    def test_metric_diff_direction(self):
        diff = MetricDiff(
            metric_name="loss", values_a=[1.0]*10, values_b=[0.5]*10,
            mean_a=1.0, mean_b=0.5, delta=-0.5, delta_pct=-50.0,
            significant=True, p_value=0.001, direction="improved",
        )
        assert "↓" in diff.summary
        assert "-50" in diff.summary
