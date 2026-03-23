from contextlib import asynccontextmanager
import asyncio
import logging
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from ml_service import config
from ml_service.drift import build_drift_monitor, run_evidently_reporting, stop_background_task
from ml_service.exceptions import (
    InvalidModelArtifactError,
    MissingFeaturesError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotReadyError,
    UnsupportedFeatureError,
)
from ml_service.features import request_payload, to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.monitoring import MetricsService
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


logger = logging.getLogger(__name__)
MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    model_container: Model = app.state.model_container
    metrics: MetricsService = app.state.metrics
    drift_monitor = app.state.drift_monitor
    stop_event = asyncio.Event()
    drift_task = asyncio.create_task(
        run_evidently_reporting(monitor=drift_monitor, metrics=metrics, stop_event=stop_event),
    )
    app.state.drift_task = drift_task
    app.state.startup_error = None

    try:
        configure_mlflow()
    except Exception as exc:
        app.state.startup_error = str(exc)
        logger.exception('Failed to configure MLflow on startup')

    default_run_id = config.default_run_id(required=False)
    if default_run_id:
        start = perf_counter()
        try:
            state = model_container.set(run_id=default_run_id)
        except Exception as exc:
            metrics.record_model_load(
                duration_seconds=perf_counter() - start,
                status='failure',
            )
            metrics.record_model_update(status='startup_failure')
            app.state.startup_error = str(exc)
            logger.exception('Failed to load default model for run_id=%s', default_run_id)
        else:
            metrics.record_model_load(
                duration_seconds=perf_counter() - start,
                status='success',
            )
            metrics.record_model_update(status='startup_success')
            metrics.set_current_model(state)
            drift_monitor.reset(run_id=state.run_id, features=state.features)
    else:
        app.state.startup_error = 'DEFAULT_RUN_ID is not set'

    try:
        yield
    finally:
        stop_event.set()
        await stop_background_task(app.state.drift_task)


def _route_pattern(request) -> str:
    route = request.scope.get('route')
    if route is None:
        return request.url.path
    return getattr(route, 'path', request.url.path)


def create_app(
    *,
    model_container: Model | None = None,
    metrics: MetricsService | None = None,
    drift_monitor=None,
) -> FastAPI:
    model_container = model_container or MODEL
    metrics = metrics or MetricsService()
    drift_monitor = drift_monitor or build_drift_monitor()

    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)
    app.state.model_container = model_container
    app.state.metrics = metrics
    app.state.drift_monitor = drift_monitor
    app.state.drift_task = None
    app.state.startup_error = None

    @app.middleware('http')
    async def metrics_middleware(request, call_next):
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.record_http_request(
                method=request.method,
                route=_route_pattern(request),
                status_code=500,
                duration_seconds=perf_counter() - start,
            )
            raise

        metrics.record_http_request(
            method=request.method,
            route=_route_pattern(request),
            status_code=response.status_code,
            duration_seconds=perf_counter() - start,
        )
        return response

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = model_container.get()
        return {
            'status': 'ok' if model_state.is_loaded else 'degraded',
            'run_id': model_state.run_id,
            'model_type': model_state.model_type,
            'features': list(model_state.features),
            'startup_error': app.state.startup_error,
        }

    @app.get('/metrics', include_in_schema=False)
    def metrics_handler() -> Response:
        return Response(content=metrics.render(), media_type=metrics.content_type())

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state = model_container.get()
        if not state.is_loaded:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        raw_payload = request_payload(request)
        metrics.record_feature_values(raw_payload)

        preprocess_start = perf_counter()
        try:
            df = to_dataframe(request, needed_columns=list(state.features))
        except MissingFeaturesError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnsupportedFeatureError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            metrics.record_preprocessing(perf_counter() - preprocess_start)

        inference_start = perf_counter()
        try:
            probability, prediction = model_container.predict(df)
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelInferenceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            metrics.record_inference(perf_counter() - inference_start)

        metrics.record_prediction(probability=probability, prediction=prediction)
        drift_monitor.record(features=df, prediction=prediction, probability=probability)
        metrics.set_pending_evidently_events(drift_monitor.pending_count())
        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id.lower()
        load_start = perf_counter()

        try:
            configure_mlflow()
            state = model_container.set(run_id=run_id)
        except InvalidModelArtifactError as exc:
            metrics.record_model_load(
                duration_seconds=perf_counter() - load_start,
                status='failure',
            )
            metrics.record_model_update(status='invalid_artifact')
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelLoadError as exc:
            metrics.record_model_load(
                duration_seconds=perf_counter() - load_start,
                status='failure',
            )
            metrics.record_model_update(status='failure')
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        metrics.record_model_load(
            duration_seconds=perf_counter() - load_start,
            status='success',
        )
        metrics.record_model_update(status='success')
        metrics.set_current_model(state)
        drift_monitor.reset(run_id=state.run_id, features=state.features)
        metrics.set_pending_evidently_events(drift_monitor.pending_count())
        app.state.startup_error = None
        return UpdateModelResponse(
            run_id=run_id,
            model_type=state.model_type or 'unknown',
            features=list(state.features),
        )

    return app


app = create_app()
