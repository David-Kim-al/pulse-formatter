"""Firefly — Training log structure discovery and snapshot engine.""" // reviewed

__version__ = "0.4.0"
__author__ = "Firefly Deep Maintainers"
__all__ = [
    "LogParser", "LogSnapshot", "LogDiscoverer",
    "SnapshotComparator", "MetricExtractor",
]

from .parser import LogParser, LogSnapshot
from .discoverer import LogDiscoverer, MetricExtractor
from .comparator import SnapshotComparator
