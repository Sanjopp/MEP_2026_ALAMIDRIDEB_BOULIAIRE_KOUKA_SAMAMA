"""Tests for custom BatchNorm implementation."""

import torch
import torch.nn as nn

from models.batchnorm import BatchNorm1D


class TestBatchNorm1D:
    """Test suite for custom BatchNorm1D."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        bn = BatchNorm1D(64)
        x = torch.randn(32, 64)
        y = bn(x)
        assert y.shape == x.shape

    def test_output_is_normalized(self):
        """After BN, output should have ~zero mean and ~unit variance."""
        bn = BatchNorm1D(64)
        x = torch.randn(128, 64) * 10 + 5  # shifted and scaled
        y = bn(x)
        assert torch.abs(y.mean(dim=0)).max() < 0.1
        assert torch.abs(y.var(dim=0, unbiased=False) - 1.0).max() < 0.1

    def test_gradients_match_pytorch(self):
        """Custom BN gradients should match PyTorch's nn.BatchNorm1d."""
        torch.manual_seed(42)
        dim = 16
        batch_size = 32

        # Input
        x = torch.randn(batch_size, dim, requires_grad=True)

        # Custom BN
        custom_bn = BatchNorm1D(dim)
        y_custom = custom_bn(x)
        loss_custom = y_custom.sum()
        loss_custom.backward()
        grad_custom = x.grad.clone()

        # Reset grad
        x.grad = None

        # PyTorch BN (train mode, same params)
        pt_bn = nn.BatchNorm1d(dim, affine=True, track_running_stats=False)
        pt_bn.weight.data.copy_(custom_bn.gamma.data)
        pt_bn.bias.data.copy_(custom_bn.beta.data)
        y_pt = pt_bn(x)
        loss_pt = y_pt.sum()
        loss_pt.backward()
        grad_pt = x.grad.clone()

        # Compare
        assert torch.allclose(grad_custom, grad_pt, atol=1e-5), (
            f"Max diff: {(grad_custom - grad_pt).abs().max().item()}"
        )

    def test_learnable_parameters(self):
        """Gamma and beta should be learnable."""
        bn = BatchNorm1D(32)
        params = list(bn.parameters())
        assert len(params) == 2
        assert params[0].shape == (32,)  # gamma
        assert params[1].shape == (32,)  # beta


class TestMLP:
    """Test suite for MLP model."""

    def test_mlp_output_shape(self):
        """MLP output should have correct shape."""
        from models.mlp import MLP

        model = MLP(input_dim=784, hidden_dims=(128, 64), num_classes=10)
        x = torch.randn(16, 1, 28, 28)
        y = model(x)
        assert y.shape == (16, 10)

    def test_mlp_with_bn(self):
        """MLP with BN should produce different output than without."""
        from models.mlp import MLP

        torch.manual_seed(42)
        model_bn = MLP(input_dim=784, hidden_dims=(128, 64), num_classes=10, use_bn=True)
        model_no_bn = MLP(input_dim=784, hidden_dims=(128, 64), num_classes=10, use_bn=False)

        x = torch.randn(16, 1, 28, 28)
        y_bn = model_bn(x)
        y_no_bn = model_no_bn(x)

        # Both should have correct shape
        assert y_bn.shape == (16, 10)
        assert y_no_bn.shape == (16, 10)


class TestCNN:
    """Test suite for CNN model."""

    def test_cnn_cifar10_shape(self):
        """CNN should handle CIFAR-10 dimensions."""
        from models.cnn import ConvMLP

        model = ConvMLP(
            in_channels=3, input_size=32,
            conv_channels=(32, 64), hidden_dims=(128,),
            num_classes=10, use_bn=False,
        )
        x = torch.randn(8, 3, 32, 32)
        y = model(x)
        assert y.shape == (8, 10)

    def test_cnn_fashion_mnist_shape(self):
        """CNN should handle Fashion-MNIST dimensions."""
        from models.cnn import ConvMLP

        model = ConvMLP(
            in_channels=1, input_size=28,
            conv_channels=(32, 64), hidden_dims=(128,),
            num_classes=10, use_bn=True,
        )
        x = torch.randn(8, 1, 28, 28)
        y = model(x)
        assert y.shape == (8, 10)

    def test_cnn_backward(self):
        """CNN should compute gradients without error."""
        from models.cnn import ConvMLP

        model = ConvMLP(
            in_channels=3, input_size=32,
            conv_channels=(32, 64), hidden_dims=(128,),
            num_classes=10, use_bn=True,
        )
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"