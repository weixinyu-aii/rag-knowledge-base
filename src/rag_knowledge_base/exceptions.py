"""Domain-specific exceptions."""


class RAGKnowledgeBaseError(Exception):
    """Base exception for the project."""


class ConfigurationError(RAGKnowledgeBaseError):
    """Raised when required configuration is invalid."""


class DocumentValidationError(RAGKnowledgeBaseError):
    """Raised when an uploaded or local document is invalid."""


class IndexNotReadyError(RAGKnowledgeBaseError):
    """Raised when retrieval is attempted before an index exists."""


class ProviderError(RAGKnowledgeBaseError):
    """Raised when an embedding or chat provider fails."""
