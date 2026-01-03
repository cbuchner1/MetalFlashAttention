"""
Basic usage example for Metal FlashAttention
"""

import torch
import time

# Check MPS availability
if not torch.backends.mps.is_available():
    print("MPS not available. This example requires Apple Silicon.")
    exit(1)

device = torch.device('mps')

# Try to import Metal FlashAttention
try:
    from metal_flash_attention import flash_attn_varlen_func, METAL_AVAILABLE
    if not METAL_AVAILABLE:
        print("Metal extension not built. Run: python setup.py build_ext --inplace")
        exit(1)
except ImportError:
    print("metal_flash_attention not found. Run: pip install -e .")
    exit(1)

print("Metal FlashAttention loaded successfully")

# Configuration
seq_len = 128
num_heads = 8
head_dim = 64
batch_size = 4

# Single sequence test
print("\n--- Single Sequence Test ---")
q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

output = flash_attn_varlen_func(q, k, v, causal=True)
print(f"Input shape: ({seq_len}, {num_heads}, {head_dim})")
print(f"Output shape: {tuple(output.shape)}")

# Batched test
print("\n--- Batched Sequences Test ---")
total_tokens = seq_len * batch_size
q_batch = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)
k_batch = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)
v_batch = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float32)

cu_seqlens = torch.tensor(
    [i * seq_len for i in range(batch_size + 1)],
    dtype=torch.int32,
    device=device
)

output_batch = flash_attn_varlen_func(
    q_batch, k_batch, v_batch,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=seq_len,
    max_seqlen_k=seq_len,
    causal=True
)
print(f"Batch size: {batch_size}")
print(f"Total tokens: {total_tokens}")
print(f"Output shape: {tuple(output_batch.shape)}")

# Benchmark
print("\n--- Performance Benchmark ---")
warmup_runs = 5
benchmark_runs = 20

for _ in range(warmup_runs):
    _ = flash_attn_varlen_func(q_batch, k_batch, v_batch,
                                cu_seqlens_q=cu_seqlens,
                                cu_seqlens_k=cu_seqlens,
                                causal=True)
torch.mps.synchronize()

start = time.perf_counter()
for _ in range(benchmark_runs):
    _ = flash_attn_varlen_func(q_batch, k_batch, v_batch,
                                cu_seqlens_q=cu_seqlens,
                                cu_seqlens_k=cu_seqlens,
                                causal=True)
torch.mps.synchronize()
elapsed = time.perf_counter() - start

avg_time_ms = (elapsed / benchmark_runs) * 1000
throughput = total_tokens / (elapsed / benchmark_runs)

print(f"Average time: {avg_time_ms:.2f} ms")
print(f"Throughput: {throughput:,.0f} tokens/second")

print("\nDone!")
