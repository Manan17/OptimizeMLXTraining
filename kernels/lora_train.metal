// MetalLoRA - Training Kernels
// Fused forward, backward, and optimizer operations for efficient training.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct TrainConfig {
    uint batch_size;
    uint seq_len;
    uint in_features;
    uint out_features;
    uint rank;
    float alpha;
    float learning_rate;
    float weight_decay;
    float grad_clip;
    float dropout_prob;
    uint accumulation_steps;
    uint current_step;
    uint seed;
};

// Forward pass with Ax caching for backward

kernel void lora_train_forward(
    device const half* x             [[buffer(0)]],
    device const half* W0            [[buffer(1)]],
    device const half* A             [[buffer(2)]],
    device const half* B             [[buffer(3)]],
    device half* output              [[buffer(4)]],
    device float* Ax_cache           [[buffer(5)]],
    constant TrainConfig& cfg        [[buffer(6)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    const uint d = tgid.x * 256 + lid;
    
    if (batch_idx >= cfg.batch_size || seq_idx >= cfg.seq_len) return;
    
    const float scale = cfg.alpha / float(cfg.rank);
    const uint x_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.in_features;
    const uint out_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.out_features;
    const uint ax_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.rank;
    
    threadgroup half tg_x[4096];
    threadgroup float tg_Ax[128];
    
    for (uint k = lid; k < cfg.in_features; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint k = 0; k < cfg.in_features; k += 4) {
            half4 a_vec = half4(A[r * cfg.in_features + k], A[r * cfg.in_features + k + 1],
                                A[r * cfg.in_features + k + 2], A[r * cfg.in_features + k + 3]);
            half4 x_vec = half4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
            acc += float(dot(a_vec, x_vec));
        }
        tg_Ax[r] = acc;
        Ax_cache[ax_offset + r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (d >= cfg.out_features) return;
    
    float h = 0.0f;
    for (uint k = 0; k < cfg.in_features; k += 4) {
        half4 w = half4(W0[d * cfg.in_features + k], W0[d * cfg.in_features + k + 1],
                        W0[d * cfg.in_features + k + 2], W0[d * cfg.in_features + k + 3]);
        half4 xv = half4(tg_x[k], tg_x[k+1], tg_x[k+2], tg_x[k+3]);
        h += float(dot(w, xv));
    }
    
    float lora = 0.0f;
    for (uint r = 0; r < cfg.rank; ++r) {
        lora += float(B[d * cfg.rank + r]) * tg_Ax[r];
    }
    
    output[out_offset + d] = half(h + scale * lora);
}

// Backward pass using cached Ax

kernel void lora_train_backward(
    device const half* grad_output   [[buffer(0)]],
    device const half* x             [[buffer(1)]],
    device const float* Ax_cache     [[buffer(2)]],
    device const half* A             [[buffer(3)]],
    device const half* B             [[buffer(4)]],
    device atomic_float* grad_A      [[buffer(5)]],
    device atomic_float* grad_B      [[buffer(6)]],
    constant TrainConfig& cfg        [[buffer(7)]],
    uint3 tgid                       [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]]
) {
    const uint batch_idx = tgid.z;
    const uint seq_idx = tgid.y;
    
    if (batch_idx >= cfg.batch_size || seq_idx >= cfg.seq_len) return;
    
    const float scale = cfg.alpha / float(cfg.rank);
    const uint x_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.in_features;
    const uint grad_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.out_features;
    const uint ax_offset = (batch_idx * cfg.seq_len + seq_idx) * cfg.rank;
    
    threadgroup half tg_x[4096];
    threadgroup half tg_grad[4096];
    threadgroup float tg_Ax[128];
    threadgroup float tg_Bt_grad[128];
    
    for (uint k = lid; k < cfg.in_features; k += 256) {
        tg_x[k] = x[x_offset + k];
    }
    for (uint d = lid; d < cfg.out_features; d += 256) {
        tg_grad[d] = grad_output[grad_offset + d];
    }
    for (uint r = lid; r < cfg.rank; r += 256) {
        tg_Ax[r] = Ax_cache[ax_offset + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint r = lid; r < cfg.rank; r += 256) {
        float acc = 0.0f;
        for (uint d = 0; d < cfg.out_features; ++d) {
            acc += float(B[d * cfg.rank + r]) * float(tg_grad[d]);
        }
        tg_Bt_grad[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint dr = lid; dr < cfg.out_features * cfg.rank; dr += 256) {
        uint d = dr / cfg.rank;
        uint r = dr % cfg.rank;
        if (d < cfg.out_features && r < cfg.rank) {
            float grad_val = clamp(scale * float(tg_grad[d]) * tg_Ax[r], -cfg.grad_clip, cfg.grad_clip);
            atomic_fetch_add_explicit(&grad_B[d * cfg.rank + r], grad_val, memory_order_relaxed);
        }
    }
    
    for (uint rk = lid; rk < cfg.rank * cfg.in_features; rk += 256) {
        uint r = rk / cfg.in_features;
        uint k = rk % cfg.in_features;
        if (r < cfg.rank && k < cfg.in_features) {
            float grad_val = clamp(scale * tg_Bt_grad[r] * float(tg_x[k]), -cfg.grad_clip, cfg.grad_clip);
            atomic_fetch_add_explicit(&grad_A[r * cfg.in_features + k], grad_val, memory_order_relaxed);
        }
    }
}

// AdamW optimizer step

kernel void lora_optimizer_step(
    device half* A                   [[buffer(0)]],
    device half* B                   [[buffer(1)]],
    device float* grad_A             [[buffer(2)]],
    device float* grad_B             [[buffer(3)]],
    device float* m_A                [[buffer(4)]],
    device float* v_A                [[buffer(5)]],
    device float* m_B                [[buffer(6)]],
    device float* v_B                [[buffer(7)]],
    constant float& learning_rate    [[buffer(8)]],
    constant float& beta1            [[buffer(9)]],
    constant float& beta2            [[buffer(10)]],
    constant float& epsilon          [[buffer(11)]],
    constant float& weight_decay     [[buffer(12)]],
    constant uint& step              [[buffer(13)]],
    constant uint& A_size            [[buffer(14)]],
    constant uint& B_size            [[buffer(15)]],
    constant uint& accum_steps       [[buffer(16)]],
    uint tid                         [[thread_position_in_grid]]
) {
    float bc1 = 1.0f - pow(beta1, float(step));
    float bc2 = 1.0f - pow(beta2, float(step));
    float lr_scaled = learning_rate * sqrt(bc2) / bc1;
    
    if (tid < A_size) {
        float grad = grad_A[tid] / float(accum_steps);
        grad_A[tid] = 0.0f;
        float param = float(A[tid]);
        
        float m = beta1 * m_A[tid] + (1.0f - beta1) * grad;
        float v = beta2 * v_A[tid] + (1.0f - beta2) * grad * grad;
        m_A[tid] = m;
        v_A[tid] = v;
        
        param -= lr_scaled * (m / (sqrt(v) + epsilon) + weight_decay * param);
        A[tid] = half(param);
    }
    
    if (tid >= A_size && tid < A_size + B_size) {
        uint idx = tid - A_size;
        float grad = grad_B[idx] / float(accum_steps);
        grad_B[idx] = 0.0f;
        float param = float(B[idx]);
        
        float m = beta1 * m_B[idx] + (1.0f - beta1) * grad;
        float v = beta2 * v_B[idx] + (1.0f - beta2) * grad * grad;
        m_B[idx] = m;
        v_B[idx] = v;
        
        param -= lr_scaled * (m / (sqrt(v) + epsilon) + weight_decay * param);
        B[idx] = half(param);
    }
}
