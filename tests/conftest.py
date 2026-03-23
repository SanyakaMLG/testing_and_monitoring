import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service.app import create_app
from ml_service.drift import DriftMonitor
from ml_service.model import Model
from ml_service.monitoring import MetricsService


class DummyPipeline:
    def __init__(
        self,
        *,
        features: list[str],
        probability: float = 0.8,
        model_type: str = 'RandomForestClassifier',
    ) -> None:
        self.feature_names_in_ = np.array(features, dtype=object)
        estimator_cls = type(model_type, (), {})
        self.steps = [('estimator', estimator_cls())]
        self._probability = probability

    def predict_proba(self, dataframe):
        if list(dataframe.columns) != list(self.feature_names_in_):
            raise AssertionError('Unexpected feature order')
        return np.array([[1.0 - self._probability, self._probability]])


def build_client(*, monkeypatch, loader_mapping: dict[str, object], default_run_id: str | None):
    def fake_load_model(*, model_uri=None, run_id=None):
        model = loader_mapping[run_id]
        if isinstance(model, Exception):
            raise model
        return model

    monkeypatch.setattr('ml_service.model.load_model', fake_load_model)
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://example.com')
    if default_run_id is None:
        monkeypatch.delenv('DEFAULT_RUN_ID', raising=False)
    else:
        monkeypatch.setenv('DEFAULT_RUN_ID', default_run_id)

    app = create_app(
        model_container=Model(),
        metrics=MetricsService(),
        drift_monitor=DriftMonitor(batch_size=10, enabled=False),
    )
    return TestClient(app)


@pytest.fixture
def predict_payload() -> dict[str, object]:
    return {
        'age': 42,
        'workclass': 'Private',
        'fnlwgt': 190000,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 0,
        'capital.loss': 0,
        'hours.per.week': 45,
        'native.country': 'United-States',
    }
