"""Application errors with user-friendly messages (no stack traces for users)."""

class AppError(Exception):
    def __init__(self, message: str, code: str = "error", status: int = 400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


class ModelNotFoundError(AppError):
    def __init__(self, which: str = ""):
        super().__init__("Trained model not found. Please train the model first.",
                         code="model_not_found", status=503)
        self.which = which


class ModelLoadError(AppError):
    def __init__(self, detail: str = ""):
        super().__init__("The trained model could not be loaded. Please re-train or check the model file.",
                         code="model_load_failed", status=503)
        self.detail = detail


class InvalidImageError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="invalid_image", status=400)


class WeatherError(AppError):
    """code is one of: no_api_key, invalid_api_key, timeout, network, location_not_found,
    api_failure, bad_response, bad_request"""
    def __init__(self, message: str, code: str, status: int = 502):
        super().__init__(message, code=code, status=status)
