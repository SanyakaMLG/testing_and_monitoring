from tests.conftest import DummyPipeline, build_client


def test_service_starts_and_serves_prediction(monkeypatch, predict_payload):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={
            'a' * 32: DummyPipeline(
                features=['age', 'education.num', 'hours.per.week'],
                probability=0.81,
            ),
        },
        default_run_id='a' * 32,
    )

    with client:
        health = client.get('/health')
        response = client.post('/predict', json=predict_payload)
        metrics = client.get('/metrics')

    assert health.status_code == 200
    assert health.json()['status'] == 'ok'
    assert response.status_code == 200
    assert response.json() == {'prediction': 1, 'probability': 0.81}
    assert metrics.status_code == 200
    assert 'ml_service_http_requests_total' in metrics.text
    assert 'ml_service_current_model_info' in metrics.text
    assert 'ml_service_prediction_total{prediction="1"} 1.0' in metrics.text


def test_predict_returns_422_when_feature_needed_by_model_is_missing(monkeypatch, predict_payload):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={
            'a' * 32: DummyPipeline(
                features=['age', 'occupation', 'hours.per.week'],
                probability=0.4,
            ),
        },
        default_run_id='a' * 32,
    )
    payload = {key: value for key, value in predict_payload.items() if key != 'occupation'}

    with client:
        response = client.post('/predict', json=payload)

    assert response.status_code == 422
    assert 'Missing required features' in response.json()['detail']
    assert 'occupation' in response.json()['detail']


def test_predict_request_validation_rejects_extra_fields(monkeypatch, predict_payload):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={
            'a' * 32: DummyPipeline(features=['age'], probability=0.4),
        },
        default_run_id='a' * 32,
    )
    payload = dict(predict_payload)
    payload['unexpected'] = 'value'

    with client:
        response = client.post('/predict', json=payload)

    assert response.status_code == 422
    assert response.json()['detail'][0]['type'] == 'extra_forbidden'


def test_update_model_returns_404_and_keeps_previous_model(monkeypatch):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={
            'a' * 32: DummyPipeline(features=['age'], probability=0.7),
            'b' * 32: RuntimeError('missing run'),
        },
        default_run_id='a' * 32,
    )

    with client:
        response = client.post('/updateModel', json={'run_id': 'b' * 32})
        health = client.get('/health')

    assert response.status_code == 404
    assert health.status_code == 200
    assert health.json()['run_id'] == 'a' * 32


def test_update_model_request_validation_rejects_bad_run_id(monkeypatch):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={
            'a' * 32: DummyPipeline(features=['age'], probability=0.7),
        },
        default_run_id='a' * 32,
    )

    with client:
        response = client.post('/updateModel', json={'run_id': 'not-a-valid-run-id'})

    assert response.status_code == 422


def test_service_starts_in_degraded_mode_without_default_model(monkeypatch):
    client = build_client(
        monkeypatch=monkeypatch,
        loader_mapping={},
        default_run_id=None,
    )

    with client:
        health = client.get('/health')
        response = client.post('/predict', json={'age': 25})

    assert health.status_code == 200
    assert health.json()['status'] == 'degraded'
    assert response.status_code == 503
