from torch import nn
import torch
from torch.autograd import Function

class ActGHFFunction(Function):
    @staticmethod
    def forward(ctx, s, t, m1, m2):

        # Forward computation
        num = 1 + m1 * t
        den = 1 + m2 * t * torch.exp(-s/t)
        output = num / den

        # Save for backward pass
        ctx.save_for_backward(t, m1, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t, m1, output = ctx.saved_tensors

        # Compute gradient using the provided formula
        grad_s = (1/t) * output * (1 - (1/(1 + m1*t)) * output)

        # Return gradients (None for fixed params t, m1, m2)
        return grad_output * grad_s, None, None, None

class ActGHF(nn.Module):
    def __init__(self, t=0.5, m1=-1.001, m2=50):
        super(ActGHF, self).__init__()
        # Register as buffers since they're fixed parameters
        self.register_buffer('t', torch.tensor(float(t)))
        self.register_buffer('m1', torch.tensor(float(m1)))
        self.register_buffer('m2', torch.tensor(float(m2)))

    def forward(self, s):
        return ActGHFFunction.apply(s, self.t, self.m1, self.m2)
