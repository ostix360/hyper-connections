import torch
from torch import nn
from torch.nn import Module

from einops import rearrange

class GRUGatedResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)
