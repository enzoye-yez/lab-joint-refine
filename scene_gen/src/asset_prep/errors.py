"""Recoverable errors; messages must not contain credentials or full service responses."""


class PrepError(Exception):
    pass


class ValidationError(PrepError):
    pass


class BudgetExceeded(PrepError):
    pass


class UncertainRequest(PrepError):
    pass


class ProviderError(PrepError):
    def __init__(self, message, *, uncertain=False, diagnostics=None):
        super().__init__(message)
        self.uncertain = uncertain
        self.diagnostics = diagnostics or {}


class StaleEvidence(PrepError):
    pass


class ReferenceUnavailable(ProviderError):
    """The service completed its response but provided no verifiable search evidence; not a credential or transport failure."""
    pass


class OutputTooLarge(PrepError):
    pass


class EvidenceRepairError(ValidationError):
    pass


class ContextLimitError(PrepError):
    """Request cannot fit; never treat this as an asset defect or retry unchanged."""
    pass


class ProviderLimitError(PrepError):
    pass
