import torch
import torch.nn as nn
from torch.autograd import Function


class BatchNormFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)

        x_hat = (x - mu) / torch.sqrt(var + eps)
        y = gamma * x_hat + beta

        ctx.save_for_backward(x_hat, gamma, var)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_hat, gamma, var = ctx.saved_tensors
        eps = ctx.eps
        N = grad_out.size(0)

        dbeta = grad_out.sum(dim=0)
        dgamma = (grad_out * x_hat).sum(dim=0)

        dxhat = grad_out * gamma
        dvar = (-0.5 * (dxhat * x_hat).sum(dim=0)) / (var + eps)
        dmu = -(dxhat / torch.sqrt(var + eps)).sum(dim=0)

        dx = (
            dxhat / torch.sqrt(var + eps)
            + dvar * 2 * x_hat / N
            + dmu / N
        )

        return dx, dgamma, dbeta, None


class BatchNorm1D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return BatchNormFunction.apply(x, self.gamma, self.beta, self.eps)
