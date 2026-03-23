import pandas as pd

from ml_service.exceptions import MissingFeaturesError, UnsupportedFeatureError
from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]
NUMERIC_FEATURES = {
    'age',
    'fnlwgt',
    'education.num',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
}
CATEGORICAL_FEATURES = set(FEATURE_COLUMNS) - NUMERIC_FEATURES


def normalize_columns(needed_columns: list[str] | None = None) -> list[str]:
    if needed_columns is None:
        return FEATURE_COLUMNS.copy()

    unsupported_columns = [
        column for column in needed_columns if column not in FEATURE_COLUMNS
    ]
    if unsupported_columns:
        raise UnsupportedFeatureError(unsupported_columns)

    return needed_columns.copy()


def request_payload(req: PredictRequest) -> dict[str, object]:
    return req.model_dump(by_alias=True)


def to_dataframe(req: PredictRequest, needed_columns: list[str] | None = None) -> pd.DataFrame:
    columns = normalize_columns(needed_columns)
    payload = request_payload(req)

    missing_columns = [column for column in columns if payload.get(column) is None]
    if missing_columns:
        raise MissingFeaturesError(missing_columns)

    row = [payload[column] for column in columns]
    return pd.DataFrame([row], columns=columns)
