"""The scaffold must fail explicitly instead of inventing predictions."""

import pytest

from app.inference import ModelNotReadyError, ModelService


def test_predict_requires_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ModelNotReadyError, match="No trained model"):
        ModelService().predict("fictional-private-message")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
