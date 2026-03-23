class ServiceError(Exception):
    """
    Base service exception used to map domain failures to HTTP responses.
    """


class MissingFeaturesError(ServiceError):
    def __init__(self, missing_features: list[str]) -> None:
        self.missing_features = missing_features
        joined = ', '.join(missing_features)
        super().__init__(f'Missing required features: {joined}')


class UnsupportedFeatureError(ServiceError):
    def __init__(self, unsupported_features: list[str]) -> None:
        self.unsupported_features = unsupported_features
        joined = ', '.join(unsupported_features)
        super().__init__(f'Model requires unsupported features: {joined}')


class ModelNotReadyError(ServiceError):
    pass


class ModelLoadError(ServiceError):
    pass


class InvalidModelArtifactError(ServiceError):
    pass


class ModelInferenceError(ServiceError):
    pass
