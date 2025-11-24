import torch


# Per-group symmetric quantization
# Splits tensor in groups, finds the max abs value and computes the scale that maps values -> int range
# With int8 (n_bits=8) the usable range is [-127, 127].
# Returns the quantized tensor and the scales
def quantize(x: torch.Tensor, n_bits: int, group_size: int):
    assert (x.numel() % group_size == 0)

    # Split tensor in groups
    x = x.reshape(-1, group_size)

    # Max int range
    int_max = 2 ** (n_bits - 1) - 1

    # Compute scale for each group
    scales = int_max / x.abs().max(dim=-1).values.unsqueeze(-1)

    # Quantize
    quant = (x * scales).round()

    return quant, scales


"""
x = torch.tensor([
    -3.2,  0.0,   1.5,   7.9,   # group 0
    -8.7, -0.1,   2.3,  10.0,   # group 1
    -15.0,  5.5, -0.5,   0.2    # group 2
], dtype=torch.float32)

quantize(x, 8, 4)

tensor([[ -51.,    0.,   24.,  127.],
        [-110.,   -1.,   29.,  127.],
        [-127.,   47.,   -4.,    2.]])
"""
