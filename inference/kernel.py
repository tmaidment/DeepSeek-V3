from typing import Tuple
import torch
import torch.nn.functional as F


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Simple CPU implementation of quantization
    x_reshaped = x.reshape(-1, block_size)
    s = torch.max(torch.abs(x_reshaped), dim=1)[0] / 448.0
    s = s.reshape(*x.size()[:-1], -1)
    y = x / s.unsqueeze(-1)
    return y.to(torch.float8_e4m3fn), s


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    # Simple CPU implementation of dequantization
    return x.to(torch.float32) * s


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    # CPU implementation using regular matrix multiplication
    return torch.matmul(a.to(torch.float32), b.to(torch.float32).t()) * (
        a_s.unsqueeze(-1) * b_s
    )
