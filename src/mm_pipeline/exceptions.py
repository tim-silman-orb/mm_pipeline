class PipelineError(Exception):
    """Base exception for expected pipeline failures."""


class ConfigurationError(PipelineError):
    """Raised when project configuration is invalid or incomplete."""


class SourceCollectionError(PipelineError):
    """Raised when a configured source cannot be collected."""


class DatasetValidationError(PipelineError):
    """Raised when the prepared master dataset is invalid."""
