#include "metal_wrapper.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <map>
#include <cstdlib>

namespace metal_flash_attention {

static bool is_verbose() {
    static int verbose = -1;
    if (verbose < 0) {
        const char* env = std::getenv("METAL_FA_VERBOSE");
        verbose = (env && std::string(env) == "1") ? 1 : 0;
    }
    return verbose == 1;
}

class BufferCache {
public:
    id<MTLBuffer> acquire(size_t size, id<MTLDevice> device) {
        for (auto it = free_buffers.begin(); it != free_buffers.end(); ++it) {
            if (it->first >= size) {
                id<MTLBuffer> buf = it->second;
                free_buffers.erase(it);
                return buf;
            }
        }
        id<MTLBuffer> new_buf = [device newBufferWithLength:size
                                                    options:MTLResourceStorageModeShared];
        if (new_buf == nil) {
            throw std::runtime_error("Failed to allocate buffer of size " + std::to_string(size));
        }
        return new_buf;
    }

    void release(id<MTLBuffer> buf) {
        if (buf != nil) {
            size_t buf_size = [buf length];
            free_buffers.insert({buf_size, buf});
        }
    }

    void clear() {
        for (auto& pair : free_buffers) {
            if (pair.second != nil) {
                [pair.second release];
            }
        }
        free_buffers.clear();
    }

    ~BufferCache() {
        clear();
    }

private:
    std::multimap<size_t, id<MTLBuffer>> free_buffers;
};

static id<MTLDevice> __unsafe_unretained g_device = nil;
static id<MTLCommandQueue> __unsafe_unretained g_queue = nil;
static id<MTLLibrary> __unsafe_unretained g_library = nil;
static id<MTLComputePipelineState> __unsafe_unretained g_pipeline = nil;
static std::string g_last_error;
static bool g_kernel_compiled = false;
static BufferCache g_buffer_cache;

const char* get_shader_source() {
    // Online softmax FlashAttention kernel
    // Each thread computes one query position for one head
    // Uses Welford's algorithm for numerically stable softmax
    static const char* shader_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void attention_forward(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    device uint* cu_seqlens_q [[buffer(4)]],
    device uint* cu_seqlens_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& causal [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& max_seqlen_q [[buffer(11)]],
    constant uint& max_seqlen_k [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_queries = cu_seqlens_q[batch_size];
    uint query_idx = gid / num_heads;
    uint head_idx = gid % num_heads;

    if (query_idx >= total_queries) {
        return;
    }

    // Find batch for this query
    uint batch_idx = 0;
    for (uint b = 0; b < batch_size; b++) {
        if (query_idx < cu_seqlens_q[b + 1]) {
            batch_idx = b;
            break;
        }
    }

    uint q_start = cu_seqlens_q[batch_idx];
    uint k_start = cu_seqlens_k[batch_idx];
    uint k_end = cu_seqlens_k[batch_idx + 1];
    uint seq_len_k = k_end - k_start;
    uint query_pos_in_seq = query_idx - q_start;

    // Online softmax
    float max_score = -1e10f;
    float sum_exp = 0.0f;
    float out_vals[128];
    for (uint d = 0; d < head_dim && d < 128; d++) {
        out_vals[d] = 0.0f;
    }

    for (uint k_local = 0; k_local < seq_len_k; k_local++) {
        uint k_idx = k_start + k_local;
        if (causal && k_local > query_pos_in_seq) { continue; }

        // Full dot product
        float score = 0.0f;
        uint q_base = query_idx * num_heads * head_dim + head_idx * head_dim;
        uint k_base = k_idx * num_heads * head_dim + head_idx * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            score += q[q_base + d] * k[k_base + d];
        }
        score *= scale;

        // Online softmax update
        float max_score_old = max_score;
        max_score = max(max_score, score);
        if (max_score > max_score_old) {
            float rescale = exp(max_score_old - max_score);
            sum_exp *= rescale;
            for (uint d = 0; d < head_dim && d < 128; d++) {
                out_vals[d] *= rescale;
            }
        }

        float attn_weight = exp(score - max_score);
        sum_exp += attn_weight;

        uint v_base = k_idx * num_heads * head_dim + head_idx * head_dim;
        for (uint d = 0; d < head_dim && d < 128; d++) {
            out_vals[d] += attn_weight * v[v_base + d];
        }
    }

    // Normalize and write output
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (uint d = 0; d < head_dim && d < 128; d++) {
            out_vals[d] *= inv_sum;
        }
    }

    uint out_base = query_idx * num_heads * head_dim + head_idx * head_dim;
    for (uint d = 0; d < head_dim && d < 128; d++) {
        output[out_base + d] = out_vals[d];
    }
}
)";
    return shader_source;
}

bool compile_kernel() {
    if (g_kernel_compiled) {
        return true;
    }

    @autoreleasepool {
        try {
            if (g_device == nil) {
                throw std::runtime_error("Device not initialized");
            }

            const char* shader_src = get_shader_source();
            NSString* shader_string = [NSString stringWithUTF8String:shader_src];

            NSError* error = nil;
            id<MTLLibrary> library = [g_device newLibraryWithSource:shader_string
                                                             options:nil
                                                               error:&error];

            if (error != nil) {
                throw std::runtime_error(std::string("Shader compilation failed: ") +
                                        [[error localizedDescription] UTF8String]);
            }

            if (library == nil) {
                throw std::runtime_error("Failed to create Metal library");
            }

            g_library = [library retain];

            id<MTLFunction> kernel_fn = [g_library newFunctionWithName:@"attention_forward"];
            if (kernel_fn == nil) {
                throw std::runtime_error("Failed to find attention_forward kernel");
            }

            error = nil;
            id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:kernel_fn
                                                                                         error:&error];

            if (error != nil) {
                throw std::runtime_error(std::string("Pipeline creation failed: ") +
                                        [[error localizedDescription] UTF8String]);
            }

            if (pipeline == nil) {
                throw std::runtime_error("Failed to create compute pipeline");
            }

            g_pipeline = [pipeline retain];
            g_kernel_compiled = true;

            if (is_verbose()) {
                std::cout << "[Metal FlashAttention] Kernel compiled successfully" << std::endl;
            }

            return true;

        } catch (const std::exception& e) {
            g_last_error = std::string("Kernel compilation: ") + e.what();
            std::cerr << "[Metal FlashAttention] " << g_last_error << std::endl;
            return false;
        }
    }
}

void dispatch_attention_kernel(
    float* q_ptr,
    float* k_ptr,
    float* v_ptr,
    float* output_ptr,
    const int* cu_seqlens_q_ptr,
    const int* cu_seqlens_k_ptr,
    int total_seq_len,
    int num_heads,
    int head_dim,
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool causal
) {
    try {
        if (!g_kernel_compiled) {
            throw std::runtime_error("Kernel not compiled");
        }

        id<MTLCommandBuffer> cmd_buffer = [g_queue commandBuffer];
        if (cmd_buffer == nil) {
            throw std::runtime_error("Failed to create command buffer");
        }

        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error("Failed to create compute encoder");
        }

        [encoder setComputePipelineState:g_pipeline];

        size_t q_size = total_seq_len * num_heads * head_dim * sizeof(float);
        size_t k_size = total_seq_len * num_heads * head_dim * sizeof(float);
        size_t v_size = total_seq_len * num_heads * head_dim * sizeof(float);
        size_t out_size = total_seq_len * num_heads * head_dim * sizeof(float);
        size_t cu_size = (batch_size + 1) * sizeof(int);

        id<MTLBuffer> q_buffer = g_buffer_cache.acquire(q_size, g_device);
        memcpy([q_buffer contents], q_ptr, q_size);

        id<MTLBuffer> k_buffer = g_buffer_cache.acquire(k_size, g_device);
        memcpy([k_buffer contents], k_ptr, k_size);

        id<MTLBuffer> v_buffer = g_buffer_cache.acquire(v_size, g_device);
        memcpy([v_buffer contents], v_ptr, v_size);

        id<MTLBuffer> cu_q_buffer = g_buffer_cache.acquire(cu_size, g_device);
        memcpy([cu_q_buffer contents], (void*)cu_seqlens_q_ptr, cu_size);

        id<MTLBuffer> cu_k_buffer = g_buffer_cache.acquire(cu_size, g_device);
        memcpy([cu_k_buffer contents], (void*)cu_seqlens_k_ptr, cu_size);

        id<MTLBuffer> output_buffer = g_buffer_cache.acquire(out_size, g_device);

        [encoder setBuffer:q_buffer offset:0 atIndex:0];
        [encoder setBuffer:k_buffer offset:0 atIndex:1];
        [encoder setBuffer:v_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBuffer:cu_q_buffer offset:0 atIndex:4];
        [encoder setBuffer:cu_k_buffer offset:0 atIndex:5];

        [encoder setBytes:&softmax_scale length:sizeof(float) atIndex:6];
        uint causal_mask = causal ? 1 : 0;
        [encoder setBytes:&causal_mask length:sizeof(uint) atIndex:7];
        uint bs = batch_size;
        [encoder setBytes:&bs length:sizeof(uint) atIndex:8];
        uint nh = num_heads;
        [encoder setBytes:&nh length:sizeof(uint) atIndex:9];
        uint hd = head_dim;
        [encoder setBytes:&hd length:sizeof(uint) atIndex:10];
        uint max_q = max_seqlen_q;
        [encoder setBytes:&max_q length:sizeof(uint) atIndex:11];
        uint max_k = max_seqlen_k;
        [encoder setBytes:&max_k length:sizeof(uint) atIndex:12];

        uint total_threads = total_seq_len * num_heads;
        uint threadgroup_size = 32;
        if (max_seqlen_q > 128) {
            threadgroup_size = 64;
        } else if (max_seqlen_q > 256) {
            threadgroup_size = 96;
        }
        threadgroup_size = (threadgroup_size > 1024) ? 1024 : threadgroup_size;

        MTLSize grid_size = MTLSizeMake(total_threads, 1, 1);
        MTLSize thread_group_size = MTLSizeMake(threadgroup_size, 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];

        if (cmd_buffer.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(std::string("GPU execution error: ") +
                                    [[cmd_buffer.error description] UTF8String]);
        }

        void* buffer_contents = [output_buffer contents];
        if (buffer_contents == nullptr) {
            throw std::runtime_error("Output buffer contents pointer is null");
        }

        memcpy(output_ptr, buffer_contents, out_size);

        g_buffer_cache.release(q_buffer);
        g_buffer_cache.release(k_buffer);
        g_buffer_cache.release(v_buffer);
        g_buffer_cache.release(cu_q_buffer);
        g_buffer_cache.release(cu_k_buffer);
        g_buffer_cache.release(output_buffer);

    } catch (const std::exception& e) {
        g_last_error = std::string("Kernel dispatch: ") + e.what();
        std::cerr << "[Metal FlashAttention] " << g_last_error << std::endl;
        throw;
    }
}

void initialize_metal_device() {
    if (g_device != nil) {
        return;
    }

    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                throw std::runtime_error("Failed to create Metal device");
            }
            g_device = [device retain];

            id<MTLCommandQueue> queue = [g_device newCommandQueue];
            if (queue == nil) {
                throw std::runtime_error("Failed to create command queue");
            }
            g_queue = [queue retain];

            if (is_verbose()) {
                std::cout << "[Metal FlashAttention] Device initialized: "
                          << [[g_device name] UTF8String] << std::endl;
            }

            if (!compile_kernel()) {
                std::cerr << "[Metal FlashAttention] Warning: Kernel compilation failed, will retry on first use"
                         << std::endl;
            }

        } catch (const std::exception& e) {
            g_last_error = std::string("Metal init: ") + e.what();
            throw;
        }
    }
}

void cleanup_metal_resources() {
    @autoreleasepool {
        g_kernel_compiled = false;
        g_buffer_cache.clear();

        if (g_pipeline != nil) {
            [g_pipeline release];
            g_pipeline = nil;
        }
        if (g_library != nil) {
            [g_library release];
            g_library = nil;
        }
        if (g_queue != nil) {
            [g_queue release];
            g_queue = nil;
        }
        if (g_device != nil) {
            [g_device release];
            g_device = nil;
        }
    }
}

}  // namespace metal_flash_attention

extern "C" {

void metal_initialize() {
    try {
        metal_flash_attention::initialize_metal_device();
    } catch (const std::exception& e) {
        metal_flash_attention::g_last_error = e.what();
        std::cerr << "[Metal FlashAttention] Init error: " << e.what() << std::endl;
    }
}

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
) {
    try {
        if (metal_flash_attention::g_device == nil) {
            metal_flash_attention::initialize_metal_device();
        }

        if (!q || !k || !v || !output) {
            throw std::runtime_error("Null input pointer");
        }
        if (total_seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
            throw std::runtime_error("Invalid tensor dimensions");
        }

        if (!metal_flash_attention::g_kernel_compiled) {
            if (!metal_flash_attention::compile_kernel()) {
                throw std::runtime_error("Kernel compilation failed");
            }
        }

        if (metal_flash_attention::is_verbose()) {
            std::cout << "[Metal FlashAttention] Forward pass:" << std::endl;
            std::cout << "  Q shape: (" << total_seq_len << ", " << num_heads << ", " << head_dim << ")" << std::endl;
            std::cout << "  Batch size: " << batch_size << std::endl;
            std::cout << "  Causal: " << (causal ? "true" : "false") << std::endl;
        }

        metal_flash_attention::dispatch_attention_kernel(
            q, k, v, output,
            cu_seqlens_q, cu_seqlens_k,
            total_seq_len, num_heads, head_dim,
            batch_size, max_seqlen_q, max_seqlen_k,
            softmax_scale, causal
        );

        return 0;

    } catch (const std::exception& e) {
        metal_flash_attention::g_last_error = std::string("Forward: ") + e.what();
        std::cerr << "[Metal FlashAttention] Error: " << metal_flash_attention::g_last_error << std::endl;
        return 1;
    }
}

void metal_cleanup() {
    @autoreleasepool {
        metal_flash_attention::cleanup_metal_resources();
        if (metal_flash_attention::is_verbose()) {
            std::cout << "[Metal FlashAttention] Cleanup completed" << std::endl;
        }
    }
}

const char* metal_get_error_message() {
    return metal_flash_attention::g_last_error.c_str();
}

}
