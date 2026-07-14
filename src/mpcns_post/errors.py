"""Public exception hierarchy."""

class MPCNSPostError(RuntimeError):
    """Base class for post-processing errors."""


class ManifestError(MPCNSPostError, ValueError):
    """Raised when a manifest is invalid or unsupported."""


class BinaryFormatError(MPCNSPostError):
    """Raised when a binary file violates the documented format."""


class ValidationError(MPCNSPostError, ValueError):
    """Raised when decoded data violates an invariant."""

