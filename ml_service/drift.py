import asyncio
import contextlib
import logging
import threading
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ml_service import config
from ml_service.monitoring import MetricsService

logger = logging.getLogger(__name__)

try:
    from evidently import Report
    from evidently.metrics import ColumnDriftMetric
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import RemoteWorkspace
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    Report = None
    ColumnDriftMetric = None
    DataDriftPreset = None
    RemoteWorkspace = None


class DriftMonitor:
    def __init__(self, *, batch_size: int, enabled: bool) -> None:
        self.batch_size = batch_size
        self.enabled = enabled
        self._lock = threading.RLock()
        self._reference_data: pd.DataFrame | None = None
        self._events: list[dict[str, Any]] = []
        self._run_id: str | None = None
        self._features: tuple[str, ...] = ()

    def reset(self, *, run_id: str | None, features: Sequence[str]) -> None:
        with self._lock:
            self._reference_data = None
            self._events = []
            self._run_id = run_id
            self._features = tuple(features)

    def record(self, *, features: pd.DataFrame, prediction: int, probability: float) -> None:
        if not self.enabled:
            return

        row = features.iloc[0].to_dict()
        row['prediction'] = int(prediction)
        row['probability'] = float(probability)

        with self._lock:
            if self._features and tuple(features.columns) != self._features:
                logger.warning('Skipping drift event with unexpected feature layout: %s', tuple(features.columns))
                return
            self._events.append(row)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._events)

    def next_batch(self) -> tuple[str | None, pd.DataFrame | None, pd.DataFrame | None]:
        with self._lock:
            if len(self._events) < self.batch_size:
                return self._run_id, None, None

            batch = pd.DataFrame(self._events[: self.batch_size])
            del self._events[: self.batch_size]

            if self._reference_data is None:
                self._reference_data = batch
                return self._run_id, None, None

            return self._run_id, self._reference_data.copy(), batch

    def restore_batch(self, batch: pd.DataFrame | None) -> None:
        if batch is None or batch.empty:
            return

        with self._lock:
            self._events = batch.to_dict(orient='records') + self._events


def build_drift_monitor() -> DriftMonitor:
    enabled = config.evidently_enabled() and Report is not None
    return DriftMonitor(
        batch_size=config.evidently_batch_size(),
        enabled=enabled,
    )


def _build_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    if Report is None or DataDriftPreset is None or ColumnDriftMetric is None:
        raise RuntimeError('Evidently is not installed')

    drift_report = Report(
        metrics=[
            DataDriftPreset(),
            ColumnDriftMetric(column_name='prediction'),
            ColumnDriftMetric(column_name='probability'),
        ],
    )
    return drift_report.run(reference_data=reference_data, current_data=current_data)


async def run_evidently_reporting(
    *,
    monitor: DriftMonitor,
    metrics: MetricsService,
    stop_event: asyncio.Event,
) -> None:
    if not monitor.enabled:
        logger.info('Evidently reporting is disabled or Evidently is not installed')
        return

    workspace = RemoteWorkspace(config.evidently_url())
    project_id = config.evidently_project_id()
    interval_seconds = config.evidently_report_interval_seconds()

    while not stop_event.is_set():
        metrics.set_pending_evidently_events(monitor.pending_count())
        run_id, reference_data, current_data = monitor.next_batch()

        if reference_data is not None and current_data is not None and project_id:
            try:
                result = _build_report(reference_data=reference_data, current_data=current_data)
                workspace.add_run(project_id, result)
                metrics.record_evidently_report(status='success')
                logger.info(
                    'Pushed Evidently report for run_id=%s with %s rows',
                    run_id,
                    len(current_data),
                )
            except Exception:
                monitor.restore_batch(current_data)
                metrics.record_evidently_report(status='failure')
                logger.exception('Failed to push Evidently report for run_id=%s', run_id)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


async def stop_background_task(task: asyncio.Task | None) -> None:
    if task is None:
        return

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
