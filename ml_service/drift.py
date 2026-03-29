import asyncio
import contextlib
import logging
import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ml_service import config
from ml_service.memory import release_process_memory
from ml_service.monitoring import MetricsService

logger = logging.getLogger(__name__)

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import RemoteWorkspace
except ImportError:
    Report = None
    DataDriftPreset = None
    RemoteWorkspace = None

try:
    from evidently.metrics import ColumnDriftMetric
except ImportError:
    ColumnDriftMetric = None

try:
    from evidently.metrics import ValueDrift
except ImportError:
    ValueDrift = None


class DriftMonitor:
    def __init__(self, *, batch_size: int, enabled: bool, max_pending_events: int = 2_000) -> None:
        self.batch_size = batch_size
        self.enabled = enabled
        self.max_pending_events = max(max_pending_events, batch_size)
        self._lock = threading.RLock()
        self._reference_data: pd.DataFrame | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_pending_events)
        self._run_id: str | None = None
        self._features: tuple[str, ...] = ()
        self._dropped_events = 0

    def reset(self, *, run_id: str | None, features: Sequence[str]) -> None:
        with self._lock:
            self._reference_data = None
            self._events = deque(maxlen=self.max_pending_events)
            self._run_id = run_id
            self._features = tuple(features)
            self._dropped_events = 0

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
            was_full = len(self._events) == self.max_pending_events
            self._events.append(row)
            if was_full:
                self._dropped_events += 1
                if self._dropped_events == 1 or self._dropped_events % 100 == 0:
                    logger.warning(
                        'Drift event buffer reached %s items; dropped %s oldest events',
                        self.max_pending_events,
                        self._dropped_events,
                    )

    def pending_count(self) -> int:
        with self._lock:
            return len(self._events)

    def has_batch(self) -> bool:
        with self._lock:
            return len(self._events) >= self.batch_size

    def next_batch(self) -> tuple[str | None, pd.DataFrame | None, pd.DataFrame | None]:
        with self._lock:
            if len(self._events) < self.batch_size:
                return self._run_id, None, None

            batch = pd.DataFrame([self._events.popleft() for _ in range(self.batch_size)])

            if self._reference_data is None:
                self._reference_data = batch
                return self._run_id, None, None

            return self._run_id, self._reference_data.copy(), batch

    def restore_batch(self, batch: pd.DataFrame | None) -> None:
        if batch is None or batch.empty:
            return

        with self._lock:
            for row in reversed(batch.to_dict(orient='records')):
                if len(self._events) == self.max_pending_events:
                    self._events.pop()
                self._events.appendleft(row)


def build_drift_monitor() -> DriftMonitor:
    enabled = config.evidently_enabled() and Report is not None
    return DriftMonitor(
        batch_size=config.evidently_batch_size(),
        enabled=enabled,
        max_pending_events=config.evidently_max_pending_events(),
    )


def _build_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    if Report is None or DataDriftPreset is None or (ColumnDriftMetric is None and ValueDrift is None):
        raise RuntimeError('Evidently is not installed')

    if ColumnDriftMetric is not None:
        prediction_metric = ColumnDriftMetric(column_name='prediction')
        probability_metric = ColumnDriftMetric(column_name='probability')
    elif ValueDrift is not None:
        prediction_metric = ValueDrift(column='prediction')
        probability_metric = ValueDrift(column='probability')
    else:
        raise RuntimeError('No compatible Evidently drift metric is available')

    drift_report = Report(
        metrics=[
            DataDriftPreset(),
            prediction_metric,
            probability_metric,
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
    max_batches_per_interval = config.evidently_max_batches_per_interval()

    while not stop_event.is_set():
        metrics.set_pending_evidently_events(monitor.pending_count())
        processed_batches = 0

        while processed_batches < max_batches_per_interval and monitor.has_batch():
            run_id, reference_data, current_data = monitor.next_batch()
            processed_batches += 1

            if reference_data is None or current_data is None:
                metrics.set_pending_evidently_events(monitor.pending_count())
                continue

            if not project_id:
                metrics.set_pending_evidently_events(monitor.pending_count())
                continue

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
                break
            finally:
                del reference_data
                del current_data
                release_process_memory(logger)

            metrics.set_pending_evidently_events(monitor.pending_count())

        release_process_memory(logger)

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
