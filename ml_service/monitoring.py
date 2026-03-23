import os
import threading
from collections.abc import Mapping
from time import time
from typing import Any

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    GCCollector,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

from ml_service.features import NUMERIC_FEATURES
from ml_service.model import ModelData

HTTP_DURATION_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
MODEL_DURATION_BUCKETS = (
    0.0005,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)
NUMERIC_VALUE_BUCKETS = (
    0.0,
    1.0,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    500.0,
    1_000.0,
    5_000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
)
PROBABILITY_BUCKETS = tuple(round(step * 0.05, 2) for step in range(21))


class MetricsService:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        GCCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        ProcessCollector(registry=self.registry, pid=lambda: os.getpid())

        self.http_requests_total = Counter(
            'ml_service_http_requests_total',
            'Total HTTP requests processed by the service',
            ['method', 'route', 'status_code'],
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            'ml_service_http_request_duration_seconds',
            'HTTP request latency in seconds',
            ['method', 'route'],
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.preprocessing_duration_seconds = Histogram(
            'ml_service_preprocessing_duration_seconds',
            'Feature preprocessing latency in seconds',
            buckets=MODEL_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.inference_duration_seconds = Histogram(
            'ml_service_inference_duration_seconds',
            'Model inference latency in seconds',
            buckets=MODEL_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.model_load_duration_seconds = Histogram(
            'ml_service_model_load_duration_seconds',
            'Model load latency in seconds',
            ['status'],
            buckets=MODEL_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.input_numeric_feature_values = Histogram(
            'ml_service_input_numeric_feature_values',
            'Distribution of numeric input feature values',
            ['feature'],
            buckets=NUMERIC_VALUE_BUCKETS,
            registry=self.registry,
        )
        self.input_categorical_feature_total = Counter(
            'ml_service_input_categorical_feature_total',
            'Counts of categorical feature values seen by the service',
            ['feature', 'value'],
            registry=self.registry,
        )
        self.input_missing_feature_total = Counter(
            'ml_service_input_missing_feature_total',
            'Counts of missing input features',
            ['feature'],
            registry=self.registry,
        )
        self.prediction_probability = Histogram(
            'ml_service_prediction_probability',
            'Distribution of model probabilities',
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.prediction_total = Counter(
            'ml_service_prediction_total',
            'Counts of model predictions',
            ['prediction'],
            registry=self.registry,
        )
        self.model_update_total = Counter(
            'ml_service_model_update_total',
            'Counts of model update attempts by status',
            ['status'],
            registry=self.registry,
        )
        self.current_model_info = Gauge(
            'ml_service_current_model_info',
            'Information about the currently loaded model',
            ['run_id', 'model_type'],
            registry=self.registry,
        )
        self.current_model_feature = Gauge(
            'ml_service_current_model_feature',
            'Features required by the currently loaded model',
            ['run_id', 'feature'],
            registry=self.registry,
        )
        self.last_model_update_timestamp_seconds = Gauge(
            'ml_service_last_model_update_timestamp_seconds',
            'Unix timestamp of the last successful model update',
            registry=self.registry,
        )
        self.evidently_report_total = Counter(
            'ml_service_evidently_report_total',
            'Counts of Evidently report generation attempts',
            ['status'],
            registry=self.registry,
        )
        self.evidently_pending_events = Gauge(
            'ml_service_evidently_pending_events',
            'Number of prediction events waiting for Evidently reporting',
            registry=self.registry,
        )
        self.last_evidently_report_timestamp_seconds = Gauge(
            'ml_service_last_evidently_report_timestamp_seconds',
            'Unix timestamp of the last successful Evidently report push',
            registry=self.registry,
        )
        self.process_cpu_percent = Gauge(
            'ml_service_process_cpu_percent',
            'Current process CPU percent',
            registry=self.registry,
        )
        self.process_memory_bytes = Gauge(
            'ml_service_process_memory_bytes',
            'Current process RSS memory in bytes',
            registry=self.registry,
        )
        self.process_thread_count = Gauge(
            'ml_service_process_thread_count',
            'Current process thread count',
            registry=self.registry,
        )

        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)
        self._lock = threading.RLock()
        self._current_model_labels: tuple[str, str] | None = None
        self._current_feature_labels: set[tuple[str, str]] = set()

    def render(self) -> bytes:
        return generate_latest(self.registry)

    @staticmethod
    def content_type() -> str:
        return CONTENT_TYPE_LATEST

    def record_http_request(self, *, method: str, route: str, status_code: int, duration_seconds: float) -> None:
        self.http_requests_total.labels(
            method=method,
            route=route,
            status_code=str(status_code),
        ).inc()
        self.http_request_duration_seconds.labels(method=method, route=route).observe(duration_seconds)
        self.update_resources()

    def record_preprocessing(self, duration_seconds: float) -> None:
        self.preprocessing_duration_seconds.observe(duration_seconds)

    def record_inference(self, duration_seconds: float) -> None:
        self.inference_duration_seconds.observe(duration_seconds)

    def record_feature_values(self, values: Mapping[str, Any]) -> None:
        for feature, value in values.items():
            if value is None:
                self.input_missing_feature_total.labels(feature=feature).inc()
                continue

            if feature in NUMERIC_FEATURES:
                try:
                    self.input_numeric_feature_values.labels(feature=feature).observe(float(value))
                except (TypeError, ValueError):
                    self.input_missing_feature_total.labels(feature=feature).inc()
                continue

            self.input_categorical_feature_total.labels(feature=feature, value=str(value)).inc()

    def record_prediction(self, *, probability: float, prediction: int) -> None:
        self.prediction_probability.observe(probability)
        self.prediction_total.labels(prediction=str(prediction)).inc()

    def record_model_load(self, *, duration_seconds: float, status: str) -> None:
        self.model_load_duration_seconds.labels(status=status).observe(duration_seconds)

    def record_model_update(self, *, status: str) -> None:
        self.model_update_total.labels(status=status).inc()

    def set_current_model(self, state: ModelData) -> None:
        if not state.run_id or not state.model_type:
            return

        with self._lock:
            if self._current_model_labels is not None:
                self.current_model_info.remove(*self._current_model_labels)

            for labels in self._current_feature_labels:
                self.current_model_feature.remove(*labels)

            self._current_model_labels = (state.run_id, state.model_type)
            self._current_feature_labels = {(state.run_id, feature) for feature in state.features}

            self.current_model_info.labels(run_id=state.run_id, model_type=state.model_type).set(1)
            for feature in state.features:
                self.current_model_feature.labels(run_id=state.run_id, feature=feature).set(1)

        self.last_model_update_timestamp_seconds.set(time())

    def record_evidently_report(self, *, status: str) -> None:
        self.evidently_report_total.labels(status=status).inc()
        if status == 'success':
            self.last_evidently_report_timestamp_seconds.set(time())

    def set_pending_evidently_events(self, count: int) -> None:
        self.evidently_pending_events.set(count)

    def update_resources(self) -> None:
        try:
            self.process_cpu_percent.set(self._process.cpu_percent(interval=None))
            self.process_memory_bytes.set(self._process.memory_info().rss)
            self.process_thread_count.set(self._process.num_threads())
        except psutil.Error:
            pass
