"""Firefly CLI — parse and compare training logs."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .parser import LogParser, LogSnapshot
from .discoverer import LogDiscoverer, MetricExtractor
from .comparator import SnapshotComparator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)


@click.group()
@click.version_option(version=__version__, prog_name="firefly")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Firefly — Training log structure discovery and visualization."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Export snapshot as JSON")
def parse(path: str, output: Optional[str]) -> None:
    """Parse a training log file into a structured snapshot."""
    parser = LogParser()
    p = Path(path)
    if p.is_dir():
        snaps = parser.parse_directory(p)
        click.echo(f"Parsed {len(snaps)} files from directory")
        for snap in snaps:
            if snap.metrics:
                click.echo(f"  {Path(snap.source_path).name}: {len(snap.metric_names)} metrics, {snap.total_steps} steps")
    else:
        snap = parser.parse(p)
        click.echo(f"Format: {snap.format.value}")
        click.echo(f"Metrics: {snap.metric_names}")
        click.echo(f"Steps: {snap.total_steps}")
        for name in snap.metric_names:
            pts = snap.range(name, end_step=5)
            vals = [f"{p.value:.4f}" for p in pts[:3]]
            click.echo(f"  {name}: {vals}")
        if output:
            export = {
                "source": snap.source_path,
                "format": snap.format.value,
                "total_steps": snap.total_steps,
                "metrics": {
                    name: [{"step": p.step, "value": p.value} for p in pts]
                    for name, pts in snap.metrics.items()
                },
            }
            Path(output).write_text(json.dumps(export, indent=2, ensure_ascii=False))
            click.echo(f"Snapshot exported to {output}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def discover(path: str) -> None:
    """Discover training experiments in a directory tree."""
    discoverer = LogDiscoverer()
    root = Path(path)
    experiments = discoverer.scan_directory(root)
    click.echo(f"Found {len(experiments)} experiment(s) in {root}")
    for exp in experiments:
        click.echo(f"\n[{exp.framework.value}] {exp.experiment_id}")
        click.echo(f"  Path: {exp.root_path}")
        click.echo(f"  Snapshots: {exp.snapshot_count}")
        click.echo(f"  Checkpoints: {len(exp.checkpoints)}")
        if exp.config:
            keys = list(exp.config.keys())[:5]
            click.echo(f"  Config keys: {keys}")


@main.command()
@click.option("-a", "--snapshot-a", type=click.Path(exists=True), required=True)
@click.option("-b", "--snapshot-b", type=click.Path(exists=True), required=True)
@click.option("--significance", type=float, default=0.05)
def compare(snapshot_a: str, snapshot_b: str, significance: float) -> None:
    """Compare two training log snapshots."""
    parser = LogParser()
    snap_a = parser.parse(Path(snapshot_a))
    snap_b = parser.parse(Path(snapshot_b))
    if not snap_a.metrics or not snap_b.metrics:
        click.echo("One or both snapshots have no metrics to compare.")
        return
    comparator = SnapshotComparator(significance=significance)
    diff = comparator.compare_snapshots(snap_a, snap_b)
    click.echo(f"\nComparing: {Path(snapshot_a).name} vs {Path(snapshot_b).name}")
    click.echo(f"Metric diffs: {len(diff.metric_diffs)}")
    click.echo(f"Improved: {len(diff.improved_metrics)}")
    click.echo(f"Regressed: {len(diff.regressed_metrics)}")
    click.echo(f"Step diff: {diff.step_diff:+d}")
    for d in diff.metric_diffs[:20]:
        sig_marker = "*" if d.significant else " "
        click.echo(f"  [{sig_marker}] {d.summary}")


if __name__ == "__main__":
    main()
