"""
Metal FlashAttention for Apple Silicon

Provides flash_attn_varlen_func as drop-in replacement for flash_attn.flash_attn_varlen_func.

Usage:
    from metal_flash_attention import flash_attn_varlen_func
    output = flash_attn_varlen_func(q, k, v, cu_seqlens_q=..., causal=True)
"""

from .metal_attn import (
    flash_attn_varlen_func,
    METAL_AVAILABLE,
    initialize,
    cleanup,
    get_error,
)

__version__ = "1.0.0"
__all__ = [
    "flash_attn_varlen_func",
    "METAL_AVAILABLE",
    "initialize",
    "cleanup",
    "get_error",
]
