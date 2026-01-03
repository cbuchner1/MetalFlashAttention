# Metal FlashAttention

Metal-accelerated FlashAttention implementation for PyTorch on Apple Silicon (M1/M2/M3/M4).

Provides a drop-in replacement for `flash_attn.flash_attn_varlen_func` that runs on Apple's Metal GPU framework.

## Performance

Benchmarked on Apple M4 Max with 8 attention heads and 64 head dimension:

| Configuration | Throughput |
|---------------|------------|
| Single sequence (1024 tokens) | 80,478 tok/s |
| Batched (4 x 128 tokens) | 199,220 tok/s |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- PyTorch 2.0+ with MPS support

## Installation

```bash
git clone https://github.com/your-repo/metal-flash-attention.git
cd metal-flash-attention
pip install -e .
```

Or build the extension directly:

```bash
python setup.py build_ext --inplace
```

## Usage

### Basic Usage

```python
import torch
from metal_flash_attention import flash_attn_varlen_func

device = torch.device('mps')

# Create tensors (total_tokens, num_heads, head_dim)
q = torch.randn(128, 8, 64, device=device, dtype=torch.float32)
k = torch.randn(128, 8, 64, device=device, dtype=torch.float32)
v = torch.randn(128, 8, 64, device=device, dtype=torch.float32)

# Single sequence
output = flash_attn_varlen_func(q, k, v, causal=True)
```

### Variable-Length Batching

```python
import torch
from metal_flash_attention import flash_attn_varlen_func

device = torch.device('mps')

# Two sequences: 30 tokens and 70 tokens
total_tokens = 100
q = torch.randn(total_tokens, 8, 64, device=device, dtype=torch.float32)
k = torch.randn(total_tokens, 8, 64, device=device, dtype=torch.float32)
v = torch.randn(total_tokens, 8, 64, device=device, dtype=torch.float32)

# Cumulative sequence lengths
cu_seqlens = torch.tensor([0, 30, 100], dtype=torch.int32, device=device)

output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=70,
    max_seqlen_k=70,
    causal=True
)
```

### Using as flash_attn Drop-in Replacement

```python
# Instead of:
# from flash_attn import flash_attn_varlen_func

# Use:
from metal_flash_attention import flash_attn_varlen_func
```

## API Reference

### flash_attn_varlen_func

```python
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
) -> torch.Tensor:
```

**Parameters:**

- `q`: Query tensor of shape `(total_q, num_heads, head_dim)`. Must be on MPS device.
- `k`: Key tensor of shape `(total_kv, num_heads, head_dim)`. Must be on MPS device.
- `v`: Value tensor of shape `(total_kv, num_heads, head_dim)`. Must be on MPS device.
- `cu_seqlens_q`: Cumulative sequence lengths for queries, shape `(batch_size + 1,)`. If None, assumes single sequence.
- `cu_seqlens_k`: Cumulative sequence lengths for keys, shape `(batch_size + 1,)`. If None, uses cu_seqlens_q.
- `max_seqlen_q`: Maximum query sequence length. If None, computed from cu_seqlens_q.
- `max_seqlen_k`: Maximum key sequence length. If None, computed from cu_seqlens_k.
- `softmax_scale`: Scaling factor for attention scores. Default: `1/sqrt(head_dim)`.
- `causal`: Whether to apply causal masking for autoregressive generation.

**Returns:**

- `output`: Attention output of shape `(total_q, num_heads, head_dim)`.

### Utility Functions

```python
from metal_flash_attention import METAL_AVAILABLE, initialize, cleanup, get_error

# Check if Metal extension is available
if METAL_AVAILABLE:
    initialize()  # Explicitly initialize Metal device
    # ... use flash_attn_varlen_func ...
    cleanup()     # Clean up Metal resources

# Get error message if something fails
error = get_error()
```

## Debug Logging

Enable verbose logging by setting the environment variable:

```bash
export METAL_FA_VERBOSE=1
python your_script.py
```

## Implementation Details

The kernel implements online softmax (Welford's algorithm) for numerical stability:

1. Each GPU thread computes one query position for one attention head
2. Iterates through all keys in the sequence
3. Uses online softmax to track running max and sum of exponentials
4. Accumulates weighted values in a single pass
5. Normalizes output at the end

Features:
- Variable-length batching via cu_seqlens format
- Optional causal masking for autoregressive models
- Buffer caching for reduced allocation overhead
- Supports head dimensions up to 128

## WIP

- Currently supports FP32 and BF16 (converted to FP32 internally)
- Paged KV cache (block_table) not yet supported
- Maximum head dimension: 128

## License

MIT License
