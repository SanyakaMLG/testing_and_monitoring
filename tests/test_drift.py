import asyncio

import pandas as pd

from ml_service.drift import DriftMonitor, run_evidently_reporting
from ml_service.monitoring import MetricsService


def _frame(age: int) -> pd.DataFrame:
    return pd.DataFrame([{'age': age}])


def test_drift_monitor_caps_queue_and_keeps_latest_events():
    monitor = DriftMonitor(batch_size=2, enabled=True, max_pending_events=3)
    monitor.reset(run_id='a' * 32, features=['age'])

    for age in range(5):
        monitor.record(features=_frame(age), prediction=age % 2, probability=age / 10)

    assert monitor.pending_count() == 3
    assert [event['age'] for event in monitor._events] == [2, 3, 4]


def test_evidently_reporting_drains_multiple_batches_per_interval(monkeypatch):
    calls: list[tuple[str, int]] = []

    class FakeWorkspace:
        def __init__(self, url):
            self.url = url

        def add_run(self, project_id, result):
            calls.append((project_id, result['rows']))

    monkeypatch.setattr('ml_service.drift.RemoteWorkspace', FakeWorkspace)
    monkeypatch.setattr('ml_service.drift._build_report', lambda *, reference_data, current_data: {'rows': len(current_data)})
    monkeypatch.setattr('ml_service.drift.config.evidently_url', lambda: 'http://example.com')
    monkeypatch.setattr('ml_service.drift.config.evidently_project_id', lambda: 'project-id')
    monkeypatch.setattr('ml_service.drift.config.evidently_report_interval_seconds', lambda: 60)
    monkeypatch.setattr('ml_service.drift.config.evidently_max_batches_per_interval', lambda: 3)

    monitor = DriftMonitor(batch_size=2, enabled=True, max_pending_events=10)
    monitor.reset(run_id='a' * 32, features=['age'])
    for age in range(6):
        monitor.record(features=_frame(age), prediction=age % 2, probability=0.1 * age)

    metrics = MetricsService()
    stop_event = asyncio.Event()

    async def exercise():
        task = asyncio.create_task(
            run_evidently_reporting(monitor=monitor, metrics=metrics, stop_event=stop_event),
        )
        await asyncio.sleep(0.05)
        stop_event.set()
        await task

    asyncio.run(exercise())

    assert calls == [('project-id', 2), ('project-id', 2)]
    assert monitor.pending_count() == 0
