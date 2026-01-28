// MetalLoRA - Quantized Kernels
// 4-bit and 8-bit quantized inference for memory-efficient deployment.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// NF4 quantization levels (optimal for normally-distributed weights)
constant float NF4_LEVELS[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f,
    -0.2844f, -0.1848f, -0.0911f, 0.0f,
    0.0796f, 0.1609f, 0.2461f, 0.3379f,
    0.4407f, 0.5627f, 0.7230f, 1.0f
};

inline float dequant_int4(uchar packed, bool high_nibble, float scale, float zero_point) {
    int val = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
    return (float(val) - zero_point) * scale;
}

inline float dequant_nf4(uchar packed, bool high_nibble, float absmax) {
    uint idx = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
    return NF4_LEVELS[idx] * absmax;
}

// QLoRA forward (INT4 base weights, FP16 LoRA)

kernel void qlora_forward_int4(
    device const uchar* W0_quant    [[buffer(0)]],
    device const float* W0_scales   [[buffer(1)]],
    device const float* W0_zeros    [[buffer(2)]],
    device const half* x            [[buffer(3)]],
    device const half* A            [[buffer(4)]],
    device const half* B            [[buffer(5)]],
    device half* out                [[buffer(6)]],
    constant uint& batch_size       [[buffer(7)]],
    constant uint& seq_len          [[buffer(8)]],
    constant uint& K                [[buffer(9)]],
    constant uint& D                [[buffer(10)]],
    constant uint& R                [[buffer(11)]],
    constant float& alpha           [[buffer(12)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const float scale = alpha / float(R);
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    
    float w_scale = W0_scales[d];
    float w_zero = W0_zeros[d];
    uint K_packed = K / 2;
    
    float h = 0.0f;
    for (uint k_packed = 0; k_packed < K_packed; ++k_packed) {
        uchar packed = W0_quant[d * K_packed + k_packed];
        uint k = k_packed * 2;
        
        float w0 = dequant_int4(packed, false, w_scale, w_zero);
        float w1 = dequant_int4(packed, true, w_scale, w_zero);
        
        h += w0 * float(x[x_offset + k]);
        if (k + 1 < K) h += w1 * float(x[x_offset + k + 1]);
    }
    
    float lora = 0.0f;
    for (uint r = 0; r < R; ++r) {
        float ax = 0.0f;
        for (uint k = 0; k < K; ++k) {
            ax += float(A[r * K + k]) * float(x[x_offset + k]);
        }
        lora += float(B[d * R + r]) * ax;
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + scale * lora);
}

// QLoRA forward (NF4 quantization)

kernel void qlora_forward_nf4(
    device const uchar* W0_quant    [[buffer(0)]],
    device const float* W0_absmax   [[buffer(1)]],
    device const half* x            [[buffer(2)]],
    device const half* A            [[buffer(3)]],
    device const half* B            [[buffer(4)]],
    device half* out                [[buffer(5)]],
    constant uint& batch_size       [[buffer(6)]],
    constant uint& seq_len          [[buffer(7)]],
    constant uint& K                [[buffer(8)]],
    constant uint& D                [[buffer(9)]],
    constant uint& R                [[buffer(10)]],
    constant float& alpha           [[buffer(11)]],
    constant uint& block_size       [[buffer(12)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= D) return;
    
    const float lora_scale = alpha / float(R);
    const uint x_offset = (batch_idx * seq_len + seq_idx) * K;
    const uint blocks_per_row = (K + block_size - 1) / block_size;
    
    float h = 0.0f;
    for (uint block = 0; block < blocks_per_row; ++block) {
        float absmax = W0_absmax[d * blocks_per_row + block];
        uint k_start = block * block_size;
        uint k_end = min(k_start + block_size, K);
        
        for (uint k = k_start; k < k_end; k += 2) {
            uint k_packed = k / 2;
            uchar packed = W0_quant[d * (K / 2) + k_packed];
            
            float w0 = dequant_nf4(packed, false, absmax);
            float w1 = (k + 1 < K) ? dequant_nf4(packed, true, absmax) : 0.0f;
            
            h += w0 * float(x[x_offset + k]);
            if (k + 1 < K) h += w1 * float(x[x_offset + k + 1]);
        }
    }
    
    float lora = 0.0f;
    for (uint r = 0; r < R; ++r) {
        float ax = 0.0f;
        for (uint k = 0; k < K; ++k) {
            ax += float(A[r * K + k]) * float(x[x_offset + k]);
        }
        lora += float(B[d * R + r]) * ax;
    }
    
    out[(batch_idx * seq_len + seq_idx) * D + d] = half(h + lora_scale * lora);
}
