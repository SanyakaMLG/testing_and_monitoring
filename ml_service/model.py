from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any

import numpy as np

from ml_service.exceptions import (
    InvalidModelArtifactError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotReadyError,
)
from ml_service.memory import release_process_memory
from ml_service.mlflow_utils import load_model


@dataclass(frozen=True)
class ModelData:
    model: Any | None
    run_id: str | None
    features: tuple[str, ...] = ()
    model_type: str | None = None
    updated_at: datetime | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


class Model:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> ModelData:
        with self.lock:
            if self.data.run_id == run_id and self.data.is_loaded:
                return self.data

        try:
            model = load_model(run_id=run_id)
        except Exception as exc:
            raise ModelLoadError(f'Failed to load model for run_id={run_id}') from exc

        features = self._extract_features(model)
        model_type = self._extract_model_type(model)
        with self.lock:
            previous_model = self.data.model
            self.data = ModelData(
                model=model,
                run_id=run_id,
                features=tuple(features),
                model_type=model_type,
                updated_at=datetime.now(timezone.utc),
            )
            if previous_model is not None:
                del previous_model
            release_process_memory()
            return self.data

    def predict(self, dataframe: Any) -> tuple[float, int]:
        state = self.get()
        if state.model is None:
            raise ModelNotReadyError('Model is not loaded yet')

        try:
            raw_probabilities = np.asarray(state.model.predict_proba(dataframe), dtype=float)
        except Exception as exc:
            raise ModelInferenceError('Model inference failed') from exc

        if raw_probabilities.ndim != 2 or raw_probabilities.shape[0] == 0 or raw_probabilities.shape[1] < 2:
            raise ModelInferenceError('Model returned invalid probability array')

        probability = float(raw_probabilities[0][1])
        if not 0.0 <= probability <= 1.0:
            raise ModelInferenceError('Model returned probability outside [0, 1]')

        prediction = int(probability >= 0.5)
        return probability, prediction

    @property
    def features(self) -> list[str]:
        return list(self.get().features)

    @property
    def model_type(self) -> str | None:
        return self.get().model_type

    @staticmethod
    def _extract_features(model: Any) -> list[str]:
        if not hasattr(model, 'predict_proba'):
            raise InvalidModelArtifactError('Loaded artifact does not implement predict_proba')

        raw_features = getattr(model, 'feature_names_in_', None)
        if raw_features is None:
            raise InvalidModelArtifactError('Loaded artifact does not expose feature_names_in_')

        features = [str(feature) for feature in raw_features]
        if not features:
            raise InvalidModelArtifactError('Loaded artifact does not contain any input features')
        return features

    @staticmethod
    def _extract_model_type(model: Any) -> str:
        steps = getattr(model, 'steps', None)
        if steps:
            return type(steps[-1][1]).__name__
        return type(model).__name__
