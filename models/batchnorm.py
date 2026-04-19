"""Custom BatchNorm1D implementation from scratch using PyTorch autograd."""

import torch
from torch.autograd import Function
import torch.nn as nn


class BatchNormFunction(Function):
    """Custom autograd function for Batch Normalization."""

    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, eps, momentum, training):
        if training:
            # Compute batch statistics
            mu = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            running_mean.data = (1 - momentum) * running_mean + momentum * mu.detach()
            running_var.data = (1 - momentum) * running_var + momentum * var.detach()
        else:
            # Use running statistics in eval mode
            mu = running_mean
            var = running_var

        std_inv = 1.0 / torch.sqrt(var + eps)
        x_centered = x - mu
        x_hat = x_centered * std_inv
        y = gamma * x_hat + beta

        ctx.save_for_backward(x_centered, x_hat, gamma, std_inv)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        _, x_hat, gamma, std_inv = ctx.saved_tensors
        n = grad_out.size(0)

        dbeta = grad_out.sum(dim=0)
        dgamma = (grad_out * x_hat).sum(dim=0)
        dxhat = grad_out * gamma
        dx = std_inv / n * (n * dxhat - dxhat.sum(dim=0) - x_hat * (dxhat * x_hat).sum(dim=0))

        return dx, dgamma, dbeta, None, None, None, None, None


class BatchNorm1D(nn.Module):
    """Batch Normalization layer for 1D inputs (fully-connected layers)."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.momentum = momentum

        # Running statistics for eval mode
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        return BatchNormFunction.apply(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training,
        )
