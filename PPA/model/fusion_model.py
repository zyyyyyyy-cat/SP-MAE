import torch
import torch.nn as nn

def interleave_features_2d(f_anon, f_orig):
    """
    f_anon, f_orig is [B, C] tensor
    Return the interleaved [B, 2*C] tensor:
      [a0, o0, a1, o1, a2, o2, ..., a(C-1), o(C-1)]
    """
    B, C = f_anon.shape
    f = torch.stack([f_anon, f_orig], dim=2)  # => shape [B, C, 2]
    f = f.view(B, 2*C)  # => [B, 2*C]
    return f


class InterleaveFusion2D(nn.Module):
    """
    interleaving the anonymized features with the original features, a fully connected layer maps them back to C dimensions.
    """
    def __init__(self, c_in, c_out=None):
        super().__init__()
        if c_out is None:
            c_out = c_in  # if want to back C，make c_out=c_in
        self.linear = nn.Linear(2*c_in, c_out, bias=False)
        self.norm = nn.LayerNorm(c_out)
        self.act = nn.ReLU()

    def forward(self, f_anon, f_orig):
        # f_anon, f_orig: [B, C]
        f_interleaved = interleave_features_2d(f_anon, f_orig)  # => [B, 2*C]
        f_fused = self.linear(f_interleaved)                    # => [B, C_out]
        #f_fused = self.norm(f_fused)
        f_fused = self.act(f_fused)
        return f_fused

