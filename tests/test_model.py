import pytest

from ml_service.exceptions import InvalidModelArtifactError, ModelInferenceError, ModelLoadError
from ml_service.model import Model

from tests.conftest import DummyPipeline


def test_model_set_is_atomic_when_next_load_fails(monkeypatch):
    loaded_models = {
        'a' * 32: DummyPipeline(features=['age', 'education.num'], probability=0.8),
        'b' * 32: RuntimeError('run not found'),
    }

    def fake_load_model(*, model_uri=None, run_id=None):
        result = loaded_models[run_id]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr('ml_service.model.load_model', fake_load_model)
    model = Model()
    initial_state = model.set('a' * 32)

    with pytest.raises(ModelLoadError):
        model.set('b' * 32)

    assert model.get() == initial_state


def test_model_set_skips_reloading_same_run_id(monkeypatch):
    calls: list[str] = []

    def fake_load_model(*, model_uri=None, run_id=None):
        calls.append(run_id)
        return DummyPipeline(features=['age'], probability=0.8)

    monkeypatch.setattr('ml_service.model.load_model', fake_load_model)
    model = Model()

    first_state = model.set('a' * 32)
    second_state = model.set('a' * 32)

    assert first_state == second_state
    assert calls == ['a' * 32]


def test_model_predict_returns_probability_and_class():
    model = Model()
    model.data = model.data.__class__(
        model=DummyPipeline(features=['age'], probability=0.73),
        run_id='a' * 32,
        features=('age',),
        model_type='RandomForestClassifier',
    )

    probability, prediction = model.predict(
        dataframe=__import__('pandas').DataFrame([{'age': 28}]),
    )

    assert probability == pytest.approx(0.73)
    assert prediction == 1


def test_model_rejects_invalid_artifact(monkeypatch):
    class InvalidModel:
        def predict_proba(self, dataframe):
            return [[0.5, 0.5]]

    monkeypatch.setattr('ml_service.model.load_model', lambda *, model_uri=None, run_id=None: InvalidModel())
    model = Model()

    with pytest.raises(InvalidModelArtifactError, match='feature_names_in_'):
        model.set('a' * 32)


def test_model_rejects_invalid_probability_output():
    class BrokenModel(DummyPipeline):
        def predict_proba(self, dataframe):
            return [[1.2]]

    model = Model()
    model.data = model.data.__class__(
        model=BrokenModel(features=['age']),
        run_id='a' * 32,
        features=('age',),
        model_type='RandomForestClassifier',
    )

    with pytest.raises(ModelInferenceError):
        model.predict(dataframe=__import__('pandas').DataFrame([{'age': 28}]))
