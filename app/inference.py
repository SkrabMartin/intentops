"""Inference boundary. Phase 0 never loads artifacts or predicts labels."""


class ModelNotReadyError(RuntimeError):
    """Raised until a trusted, trained model is implemented in a later phase."""


def predict(text: str) -> None:
    """Reject inference explicitly without logging or persisting customer text."""
    raise ModelNotReadyError("No trained model is available.")
