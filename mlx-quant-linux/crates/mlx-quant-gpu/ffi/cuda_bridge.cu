#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "../shaders/ternary_beehive.cu"

namespace {

int g_cuda_device = -1;
int g_cc_major = 0;
int g_cc_minor = 0;
char g_device_name[256] = {0};

int translate_cuda_err(cudaError_t err) {
    return (err == cudaSuccess) ? 0 : -8;
}

uint64_t elapsed_ms_to_ns(float ms) {
    if (ms <= 0.0f) {
        return 0ULL;
    }
    return static_cast<uint64_t>(ms * 1000000.0);
}

} // namespace

extern "C" int mq_cuda_init(
    char* device_name_buf,
    size_t name_cap,
    int* cc_major_out,
    int* cc_minor_out
) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        return -2;
    }

    int dev = 0;
    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        return -2;
    }

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        return -2;
    }

    g_cuda_device = dev;
    g_cc_major = prop.major;
    g_cc_minor = prop.minor;
    snprintf(g_device_name, sizeof(g_device_name), "%s", prop.name);

    if (device_name_buf != nullptr && name_cap > 0) {
        snprintf(device_name_buf, name_cap, "%s", g_device_name);
    }
    if (cc_major_out != nullptr) {
        *cc_major_out = g_cc_major;
    }
    if (cc_minor_out != nullptr) {
        *cc_minor_out = g_cc_minor;
    }
    return 0;
}

extern "C" int mq_cuda_ternary_2bit_gemv(
    const uint8_t* w_packed,
    size_t w_bytes,
    const int8_t* x,
    uint32_t rows,
    uint32_t cols,
    int32_t* y_out,
    uint64_t* gpu_ns_out
) {
    if (g_cuda_device < 0) {
        return -5;
    }
    if (rows == 0) {
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = 0;
        }
        return 0;
    }
    if (y_out == nullptr || (w_bytes > 0 && w_packed == nullptr) || (cols > 0 && x == nullptr)) {
        return -1;
    }

    const size_t safe_w_bytes = (w_bytes > 0) ? w_bytes : 4;
    const size_t x_bytes = (cols > 0) ? static_cast<size_t>(cols) : 4;
    const size_t y_bytes = static_cast<size_t>(rows) * sizeof(int32_t);

    uint8_t* d_w = nullptr;
    int8_t* d_x = nullptr;
    int32_t* d_y = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_w, safe_w_bytes) != cudaSuccess ||
        cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_y, y_bytes) != cudaSuccess) {
        rc = -6;
        goto cleanup;
    }

    if (w_bytes > 0) {
        if (cudaMemcpy(d_w, w_packed, w_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    } else {
        cudaMemset(d_w, 0, safe_w_bytes);
    }
    if (cols > 0) {
        if (cudaMemcpy(d_x, x, cols, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    } else {
        cudaMemset(d_x, 0, x_bytes);
    }

    if (cudaEventCreate(&ev_start) != cudaSuccess || cudaEventCreate(&ev_stop) != cudaSuccess) {
        rc = -7;
        goto cleanup;
    }

    {
        const uint32_t threads_per_block = 256u; // 8 warps/block -> 8 rows/block
        const uint32_t total_threads = rows * 32u;
        const uint32_t blocks = (total_threads + threads_per_block - 1u) / threads_per_block;

        cudaEventRecord(ev_start, 0);
        ternary_2bit_gemv_i8<<<blocks, threads_per_block>>>(d_w, d_x, d_y, rows, cols);
        cudaEventRecord(ev_stop, 0);

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            rc = translate_cuda_err(launch_err);
            goto cleanup;
        }
        const cudaError_t sync_err = cudaEventSynchronize(ev_stop);
        if (sync_err != cudaSuccess) {
            rc = translate_cuda_err(sync_err);
            goto cleanup;
        }

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = elapsed_ms_to_ns(elapsed_ms);
        }

        if (cudaMemcpy(y_out, d_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    }

cleanup:
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
    if (d_w != nullptr) cudaFree(d_w);
    if (d_x != nullptr) cudaFree(d_x);
    if (d_y != nullptr) cudaFree(d_y);
    return rc;
}

extern "C" int mq_cuda_bitcos_gemv(
    const uint8_t* presence,
    size_t p_bytes,
    const uint8_t* signs,
    size_t s_bytes,
    const uint32_t* row_sign_byte_offsets,
    const int8_t* x,
    uint32_t rows,
    uint32_t cols,
    int32_t* y_out,
    uint64_t* gpu_ns_out
) {
    if (g_cuda_device < 0) {
        return -5;
    }
    if (rows == 0) {
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = 0;
        }
        return 0;
    }
    if (row_sign_byte_offsets == nullptr || y_out == nullptr ||
        (p_bytes > 0 && presence == nullptr) ||
        (s_bytes > 0 && signs == nullptr) ||
        (cols > 0 && x == nullptr)) {
        return -1;
    }

    const size_t safe_p = (p_bytes > 0) ? p_bytes : 4;
    const size_t safe_s = (s_bytes > 0) ? s_bytes : 4;
    const size_t off_bytes = static_cast<size_t>(rows) * sizeof(uint32_t);
    const size_t x_bytes = (cols > 0) ? static_cast<size_t>(cols) : 4;
    const size_t y_bytes = static_cast<size_t>(rows) * sizeof(int32_t);

    uint8_t* d_p = nullptr;
    uint8_t* d_s = nullptr;
    uint32_t* d_off = nullptr;
    int8_t* d_x = nullptr;
    int32_t* d_y = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_p, safe_p) != cudaSuccess ||
        cudaMalloc(&d_s, safe_s) != cudaSuccess ||
        cudaMalloc(&d_off, off_bytes) != cudaSuccess ||
        cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_y, y_bytes) != cudaSuccess) {
        rc = -6;
        goto cleanup;
    }

    if (p_bytes > 0) {
        if (cudaMemcpy(d_p, presence, p_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    } else {
        cudaMemset(d_p, 0, safe_p);
    }
    if (s_bytes > 0) {
        if (cudaMemcpy(d_s, signs, s_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    } else {
        cudaMemset(d_s, 0, safe_s);
    }
    if (cudaMemcpy(d_off, row_sign_byte_offsets, off_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = -8;
        goto cleanup;
    }
    if (cols > 0) {
        if (cudaMemcpy(d_x, x, cols, cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    } else {
        cudaMemset(d_x, 0, x_bytes);
    }

    if (cudaEventCreate(&ev_start) != cudaSuccess || cudaEventCreate(&ev_stop) != cudaSuccess) {
        rc = -7;
        goto cleanup;
    }

    {
        const uint32_t threads_per_block = 256u;
        const uint32_t total_threads = rows * 32u;
        const uint32_t blocks = (total_threads + threads_per_block - 1u) / threads_per_block;

        cudaEventRecord(ev_start, 0);
        bitcos_gemv_i8<<<blocks, threads_per_block>>>(d_p, d_s, d_off, d_x, d_y, rows, cols);
        cudaEventRecord(ev_stop, 0);

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            rc = translate_cuda_err(launch_err);
            goto cleanup;
        }
        const cudaError_t sync_err = cudaEventSynchronize(ev_stop);
        if (sync_err != cudaSuccess) {
            rc = translate_cuda_err(sync_err);
            goto cleanup;
        }

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = elapsed_ms_to_ns(elapsed_ms);
        }

        if (cudaMemcpy(y_out, d_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    }

cleanup:
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
    if (d_p != nullptr) cudaFree(d_p);
    if (d_s != nullptr) cudaFree(d_s);
    if (d_off != nullptr) cudaFree(d_off);
    if (d_x != nullptr) cudaFree(d_x);
    if (d_y != nullptr) cudaFree(d_y);
    return rc;
}

extern "C" int mq_cuda_tequila_ste_q16(
    int32_t* weights_q16,
    const int32_t* grads_q16,
    uint32_t len,
    int32_t delta_q16,
    int32_t lr_q16,
    int32_t lambda_q16,
    uint64_t* gpu_ns_out
) {
    if (g_cuda_device < 0) {
        return -5;
    }
    if (len == 0) {
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = 0;
        }
        return 0;
    }
    if (weights_q16 == nullptr || grads_q16 == nullptr) {
        return -1;
    }

    const size_t bytes = static_cast<size_t>(len) * sizeof(int32_t);
    int32_t* d_w = nullptr;
    int32_t* d_g = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    int rc = 0;

    if (cudaMalloc(&d_w, bytes) != cudaSuccess || cudaMalloc(&d_g, bytes) != cudaSuccess) {
        rc = -6;
        goto cleanup;
    }
    if (cudaMemcpy(d_w, weights_q16, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_g, grads_q16, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = -8;
        goto cleanup;
    }
    if (cudaEventCreate(&ev_start) != cudaSuccess || cudaEventCreate(&ev_stop) != cudaSuccess) {
        rc = -7;
        goto cleanup;
    }

    {
        const uint32_t threads_per_block = 256u;
        const uint32_t blocks = (len + threads_per_block - 1u) / threads_per_block;

        cudaEventRecord(ev_start, 0);
        tequila_deadzone_ste_q16<<<blocks, threads_per_block>>>(
            d_w, d_g, len, delta_q16, lr_q16, lambda_q16);
        cudaEventRecord(ev_stop, 0);

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            rc = translate_cuda_err(launch_err);
            goto cleanup;
        }
        const cudaError_t sync_err = cudaEventSynchronize(ev_stop);
        if (sync_err != cudaSuccess) {
            rc = translate_cuda_err(sync_err);
            goto cleanup;
        }

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        if (gpu_ns_out != nullptr) {
            *gpu_ns_out = elapsed_ms_to_ns(elapsed_ms);
        }

        if (cudaMemcpy(weights_q16, d_w, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            rc = -8;
            goto cleanup;
        }
    }

cleanup:
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
    if (d_w != nullptr) cudaFree(d_w);
    if (d_g != nullptr) cudaFree(d_g);
    return rc;
}
