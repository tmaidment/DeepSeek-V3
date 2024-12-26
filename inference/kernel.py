from typing import Tuple
import torch
import torch.nn.functional as F


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0

    # Reshape to match the block structure
    x_reshaped = x.view(-1, block_size)

    # Calculate scale (maximum absolute value / 448.0) for each block
    s = torch.max(torch.abs(x_reshaped), dim=1)[0] / 448.0

    # Reshape scale back to match input dimensions
    s = s.view(*x.size()[:-1], x.size(-1) // block_size)

    # Expand scale to match input shape for division
    s_expanded = s.unsqueeze(-1).expand(*x.size()[:-1], -1, block_size).reshape(x.shape)

    # Quantize
    y = (x / s_expanded).to(torch.float8_e4m3fn)

    return y, s


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()

    # Convert to float32 first
    x_float = x.to(torch.float32)

    # Expand scale to match weight dimensions
    s_expanded = s.repeat_interleave(block_size, dim=0)[:M].repeat_interleave(
        block_size, dim=1
    )[:, :N]

    # Dequantize
    return x_float * s_expanded


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    # Convert inputs to float32
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)

    # Perform matrix multiplication
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # Reshape inputs if needed
    a_reshaped = a_float.view(M, K)

    # Compute the matrix multiplication with scales
    # Expand scales to match the multiplication dimensions
    a_s_expanded = a_s.view(-1, 1)  # Shape: (M, 1)
    b_s_expanded = b_s.view(1, -1)  # Shape: (1, N//block_size)

    # Perform the scaled matrix multiplication
    c = torch.matmul(a_reshaped, b_float.t())
    c = c * a_s_expanded * b_s_expanded.expand(-1, N)

    # Reshape result back to original dimensions
    return c.view(*a.size()[:-1], N)
