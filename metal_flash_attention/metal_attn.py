"""
Metal FlashAttention Python Wrapper

Provides flash_attn_varlen_func compatible interface that uses Metal GPU acceleration.
"""

import torch
from typing import Optional

try:
    import metal_flash_attn
    METAL_AVAILABLE = True
    _METAL_ERROR = None
except ImportError as e:
    METAL_AVAILABLE = False
    _METAL_ERROR = str(e)
    metal_flash_attn = None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Metal-accelerated FlashAttention for variable-length sequences.

    Drop-in replacement for flash_attn.flash_attn_varlen_func with Metal GPU support.

    Args:
        q: Query tensor of shape (total_q, num_heads, head_dim)
        k: Key tensor of shape (total_kv, num_heads, head_dim)
        v: Value tensor of shape (total_kv, num_heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys (batch_size + 1,)
        max_seqlen_q: Maximum query sequence length in batch
        max_seqlen_k: Maximum key sequence length in batch
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        block_table: Block table for paged KV cache (not yet supported)

    Returns:
        output: Attention output of shape (total_q, num_heads, head_dim)
    """

    if not METAL_AVAILABLE:
        raise ImportError(
            f"Metal FlashAttention not available. Error: {_METAL_ERROR}\n"
            "Build with: python setup.py build_ext --inplace"
        )

    if q.device.type != 'mps':
        raise RuntimeError(f"Q must be on MPS device, got: {q.device}")
    if k.device.type != 'mps':
        raise RuntimeError(f"K must be on MPS device, got: {k.device}")
    if v.device.type != 'mps':
        raise RuntimeError(f"V must be on MPS device, got: {v.device}")

    if q.dtype not in [torch.float32, torch.bfloat16]:
        raise RuntimeError(f"Q must be float32 or bfloat16, got: {q.dtype}")

    if q.dim() != 3:
        raise RuntimeError(f"Q must be 3D, got shape: {q.shape}")
    if k.dim() != 3:
        raise RuntimeError(f"K must be 3D, got shape: {k.shape}")
    if v.dim() != 3:
        raise RuntimeError(f"V must be 3D, got shape: {v.shape}")

    if q.size(1) != k.size(1) or q.size(1) != v.size(1):
        raise RuntimeError("Q, K, V must have same number of heads")
    if q.size(2) != k.size(2) or q.size(2) != v.size(2):
        raise RuntimeError("Q, K, V must have same head dimension")

    head_dim = q.size(2)

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    if cu_seqlens_q is None:
        total_seq = q.size(0)
        cu_seqlens_q = torch.tensor([0, total_seq], dtype=torch.int32, device=q.device)

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

    if max_seqlen_q is None:
        max_seqlen_q = int(torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item())
    if max_seqlen_k is None:
        max_seqlen_k = int(torch.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1]).item())

    if block_table is not None:
        import warnings
        warnings.warn(
            "block_table (paged KV cache) is not yet supported in Metal FlashAttention.",
            RuntimeWarning
        )

    output = metal_flash_attn.flash_attn_varlen_forward(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, causal
    )

    return output


def initialize():
    """Explicitly initialize Metal device"""
    if METAL_AVAILABLE:
        metal_flash_attn.initialize()


def cleanup():
    """Clean up Metal resources"""
    if METAL_AVAILABLE:
        metal_flash_attn.cleanup()


def get_error():
    """Get last error message from Metal"""
    if METAL_AVAILABLE:
        return metal_flash_attn.get_error()
    return _METAL_ERROR


__all__ = [
    "flash_attn_varlen_func",
    "METAL_AVAILABLE",
    "initialize",
    "cleanup",
    "get_error",
]
