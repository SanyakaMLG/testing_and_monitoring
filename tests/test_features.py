import pytest

from ml_service.exceptions import MissingFeaturesError, UnsupportedFeatureError
from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_uses_only_needed_columns_and_aliases():
    request = PredictRequest(
        age=33,
        workclass='Private',
        fnlwgt=120000,
        education='Masters',
        **{
            'education.num': 14,
            'hours.per.week': 40,
        },
    )

    dataframe = to_dataframe(
        request,
        needed_columns=['education.num', 'age', 'hours.per.week'],
    )

    assert dataframe.columns.tolist() == ['education.num', 'age', 'hours.per.week']
    assert dataframe.iloc[0].to_dict() == {
        'education.num': 14,
        'age': 33,
        'hours.per.week': 40,
    }


def test_to_dataframe_raises_when_required_feature_is_missing():
    request = PredictRequest(
        workclass='Private',
        **{
            'education.num': 14,
        },
    )

    with pytest.raises(MissingFeaturesError, match='age'):
        to_dataframe(request, needed_columns=['age', 'education.num'])


def test_to_dataframe_rejects_unknown_model_features():
    request = PredictRequest(age=33)

    with pytest.raises(UnsupportedFeatureError, match='unknown_feature'):
        to_dataframe(request, needed_columns=['age', 'unknown_feature'])
