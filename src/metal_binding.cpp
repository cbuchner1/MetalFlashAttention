#include "metal_wrapper.h"
#include <torch/extension.h>
#include <iostream>
#include <cstdlib>

static bool is_verbose() {
    static int verbose = -1;
    if (verbose < 0) {
        const char* env = std::getenv("METAL_FA_VERBOSE");
        verbose = (env && std::string(env) == "1") ? 1 : 0;
    }
    return verbose == 1;
}

torch::Tensor metal_flash_attn_varlen_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool causal
) {
    if (!q.device().is_mps()) {
        throw std::runtime_error("Q must be on MPS device");
    }
    if (!k.device().is_mps()) {
        throw std::runtime_error("K must be on MPS device");
    }
    if (!v.device().is_mps()) {
        throw std::runtime_error("V must be on MPS device");
    }

    if (q.scalar_type() != torch::kFloat32 && q.scalar_type() != torch::kBFloat16) {
        throw std::runtime_error(
            "Q must be FP32 or BF16, got: " + std::string(torch::toString(q.scalar_type()))
        );
    }

    TORCH_CHECK(q.dim() == 3, "Q must be 3D: (seq_len, num_heads, head_dim)");
    TORCH_CHECK(k.dim() == 3, "K must be 3D");
    TORCH_CHECK(v.dim() == 3, "V must be 3D");
    TORCH_CHECK(q.size(1) == k.size(1), "Q and K must have same num_heads");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V must have same num_heads");
    TORCH_CHECK(q.size(2) == k.size(2), "Q and K must have same head_dim");
    TORCH_CHECK(k.size(2) == v.size(2), "K and V must have same head_dim");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");
    TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1D");
    TORCH_CHECK(cu_seqlens_k.dim() == 1, "cu_seqlens_k must be 1D");

    int total_seq_q = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int batch_size = cu_seqlens_q.size(0) - 1;

    if (is_verbose()) {
        std::cout << "[Metal FlashAttention] PyTorch wrapper called:" << std::endl;
        std::cout << "  Q: (" << total_seq_q << ", " << num_heads << ", " << head_dim << ")" << std::endl;
        std::cout << "  Batch size: " << batch_size << ", Causal: " << (causal ? "yes" : "no") << std::endl;
    }

    // Move tensors to CPU for Metal processing
    torch::Tensor q_cpu = q.to(torch::kCPU).contiguous();
    torch::Tensor k_cpu = k.to(torch::kCPU).contiguous();
    torch::Tensor v_cpu = v.to(torch::kCPU).contiguous();
    torch::Tensor cu_seqlens_q_cpu = cu_seqlens_q.to(torch::kCPU).contiguous();
    torch::Tensor cu_seqlens_k_cpu = cu_seqlens_k.to(torch::kCPU).contiguous();
    torch::Tensor output_cpu = torch::empty_like(q_cpu);

    float* q_ptr = q_cpu.data_ptr<float>();
    float* k_ptr = k_cpu.data_ptr<float>();
    float* v_ptr = v_cpu.data_ptr<float>();
    int* cu_seqlens_q_ptr = cu_seqlens_q_cpu.data_ptr<int>();
    int* cu_seqlens_k_ptr = cu_seqlens_k_cpu.data_ptr<int>();
    float* output_ptr = output_cpu.data_ptr<float>();

    try {
        metal_initialize();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Metal initialization failed: ") + e.what()
        );
    }

    int ret = metal_attention_forward(
        q_ptr,
        k_ptr,
        v_ptr,
        total_seq_q,
        num_heads,
        head_dim,
        cu_seqlens_q_ptr,
        cu_seqlens_k_ptr,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        output_ptr
    );

    if (ret != 0) {
        throw std::runtime_error(
            std::string("Metal attention failed: ") + metal_get_error_message()
        );
    }

    return output_cpu.to(q.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Metal FlashAttention for Apple Silicon";

    m.def(
        "flash_attn_varlen_forward",
        &metal_flash_attn_varlen_forward,
        "Metal-accelerated FlashAttention for variable-length sequences",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("cu_seqlens_q"),
        pybind11::arg("cu_seqlens_k"),
        pybind11::arg("max_seqlen_q"),
        pybind11::arg("max_seqlen_k"),
        pybind11::arg("softmax_scale") = 0.125f,
        pybind11::arg("causal") = true
    );

    m.def("initialize", &metal_initialize, "Initialize Metal device");
    m.def("cleanup", &metal_cleanup, "Clean up Metal resources");
    m.def("get_error", &metal_get_error_message, "Get last error message");
}
