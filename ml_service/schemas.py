from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, str_strip_whitespace=True)

    age: int | None = Field(default=None, ge=0, le=120, description='Возраст человека')
    workclass: str | None = Field(default=None, description='Тип занятости')
    fnlwgt: int | None = Field(
        default=None,
        ge=0,
        description='Вес наблюдения в данных переписи',
    )
    education: str | None = Field(default=None, description='Образование')
    education_num: int | None = Field(
        default=None,
        alias='education.num',
        ge=0,
        description='Уровень образования в виде числа',
    )
    marital_status: str | None = Field(
        default=None,
        alias='marital.status',
        description='Семейное положение',
    )
    occupation: str | None = Field(
        default=None,
        description='Профессия / род деятельности',
    )
    relationship: str | None = Field(
        default=None,
        description='Роль человека в семье',
    )
    race: str | None = Field(default=None, description='Расовая группа')
    sex: str | None = Field(
        default=None,
        description='Пол человека (Male / Female)',
    )
    capital_gain: int | None = Field(
        default=None,
        alias='capital.gain',
        ge=0,
        description='Доход от капитала (прибыль от продажи активов)',
    )
    capital_loss: int | None = Field(
        default=None,
        alias='capital.loss',
        ge=0,
        description='Убытки от капитала',
    )
    hours_per_week: int | None = Field(
        default=None,
        alias='hours.per.week',
        ge=0,
        le=168,
        description='Количество рабочих часов в неделю',
    )
    native_country: str | None = Field(
        default=None,
        alias='native.country',
        description='Страна происхождения',
    )


class PredictResponse(BaseModel):
    prediction: int = Field(description='Предсказанный класс')
    probability: float = Field(ge=0.0, le=1.0, description='Вероятность положительного класса')


class UpdateModelRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    run_id: str = Field(min_length=32, max_length=32, pattern=r'^[0-9a-fA-F]{32}$', description='MLflow run_id')


class UpdateModelResponse(BaseModel):
    run_id: str = Field(description='MLflow run_id')
    model_type: str = Field(description='Тип загруженной модели')
    features: list[str] = Field(description='Признаки, необходимые текущей модели')
