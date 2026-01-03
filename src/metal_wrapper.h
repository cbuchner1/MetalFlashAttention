#ifndef METAL_WRAPPER_H
#define METAL_WRAPPER_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void metal_initialize();

int metal_attention_forward(
    float *q,
    float *k,
    float *v,
    int total_seq_len,
    int num_heads,
    int head_dim,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool causal,
    float *output
);

void metal_cleanup();

const char* metal_get_error_message();

#ifdef __cplusplus
}
#endif

#endif // METAL_WRAPPER_H
