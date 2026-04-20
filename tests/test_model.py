import torch
from src.model.architecture import ReadmissionPredictor


def test_model_creates_with_correct_output_shape():
    """Model should output one value per sample."""
    model = ReadmissionPredictor(input_dim=40, hidden_dims=[64, 32], dropout_rate=0.2)
    X = torch.randn(10, 40)   # batch of 10 samples, 40 features
    output = model(X)
    assert output.shape == (10,), f"Expected (10,), got {output.shape}"


def test_model_output_is_not_nan():
    """Model output should not contain NaN values."""
    model = ReadmissionPredictor(input_dim=40)
    X = torch.randn(5, 40)
    output = model(X)
    assert not torch.isnan(output).any(), "Model produced NaN values"


def test_model_accepts_different_input_dims():
    """Model should work with any input dimension."""
    for dim in [10, 20, 50, 100]:
        model = ReadmissionPredictor(input_dim=dim)
        X = torch.randn(3, dim)
        output = model(X)
        assert output.shape == (3,)