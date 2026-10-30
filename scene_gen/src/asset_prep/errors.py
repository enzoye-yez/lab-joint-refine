"""可恢复错误；信息中不得包含凭据或完整服务响应。"""


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
    """服务已完成响应，但未提供可验证检索证据；不是凭据或传输失败。"""
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
