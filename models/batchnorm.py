"""Custom BatchNorm1D implementation from scratch using PyTorch autograd."""

import torch
import torch.nn as nn
from torch.autograd import Function


class BatchNormFunction(Function):
    """Custom autograd function for Batch Normalization."""

    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std_inv = 1.0 / torch.sqrt(var + eps)

        x_centered = x - mu
        x_hat = x_centered * std_inv
        y = gamma * x_hat + beta

        # Save tensors needed for backward
        ctx.save_for_backward(x_centered, x_hat, gamma, std_inv)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_centered, x_hat, gamma, std_inv = ctx.saved_tensors
        N = grad_out.size(0)

        # Gradients for gamma and beta
        dbeta = grad_out.sum(dim=0)
        dgamma = (grad_out * x_hat).sum(dim=0)

        # Gradient for x (corrected formula)
        dxhat = grad_out * gamma
        dx = (
            std_inv
            / N
            * (
                N * dxhat
                - dxhat.sum(dim=0)
                - x_hat * (dxhat * x_hat).sum(dim=0)
            )
        )

        return dx, dgamma, dbeta, None


class BatchNorm1D(nn.Module):
    """Batch Normalization layer for 1D inputs (fully-connected layers)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return BatchNormFunction.apply(x, self.gamma, self.beta, self.eps)