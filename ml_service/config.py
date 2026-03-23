import os

MODEL_ARTIFACT_PATH = 'model'
DEFAULT_EVIDENTLY_URL = 'http://158.160.2.37:8000/'
DEFAULT_EVIDENTLY_PROJECT_ID = '019d061f-cc08-7b5e-b932-d792a1f258e2'
DEFAULT_APP_PORT = 8890


def _get_env(name: str, *, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f'Please set {name}')
    return value


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f'Environment variable {name} must be an integer') from exc


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def tracking_uri(*, required: bool = True) -> str | None:
    return _get_env('MLFLOW_TRACKING_URI', required=required)


def default_run_id(*, required: bool = True) -> str | None:
    """
    Returns the default run_id for startup.
    """

    return _get_env('DEFAULT_RUN_ID', required=required)


def app_port() -> int:
    return _get_int('APP_PORT', DEFAULT_APP_PORT)


def evidently_enabled() -> bool:
    return _get_bool('EVIDENTLY_ENABLED', True)


def evidently_url() -> str | None:
    return _get_env('EVIDENTLY_URL', default=DEFAULT_EVIDENTLY_URL)


def evidently_project_id() -> str | None:
    return _get_env('EVIDENTLY_PROJECT_ID', default=DEFAULT_EVIDENTLY_PROJECT_ID)


def evidently_batch_size() -> int:
    return _get_int('EVIDENTLY_BATCH_SIZE', 100)


def evidently_report_interval_seconds() -> int:
    return _get_int('EVIDENTLY_REPORT_INTERVAL_SECONDS', 300)
