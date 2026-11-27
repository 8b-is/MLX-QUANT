// Objective-C + ARC Metal runtime bridge for MLX-QUANT on macOS.
// Must compile cleanly with: xcrun clang -fobjc-arc -O2 -Wall -Wextra -arch arm64 -c

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    MQ_OK = 0,
    MQ_ERR_INVALID_ARG = -1,
    MQ_ERR_NO_DEVICE = -2,
    MQ_ERR_LIBRARY_COMPILE = -3,
    MQ_ERR_PIPELINE_CREATE = -4,
    MQ_ERR_NOT_INITIALIZED = -5,
    MQ_ERR_BUFFER_ALLOC = -6,
    MQ_ERR_CMD_ENCODE = -7,
    MQ_ERR_EXECUTION = -8,
};

static pthread_mutex_t g_metal_mutex = PTHREAD_MUTEX_INITIALIZER;
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_pso_ternary = nil;
static id<MTLComputePipelineState> g_pso_bitcos = nil;
static id<MTLComputePipelineState> g_pso_tequila = nil;
static id<MTLComputePipelineState> g_pso_sinkhorn = nil;
static id<MTLComputePipelineState> g_pso_engram = nil;
static BOOL g_initialized = NO;

static id<MTLComputePipelineState> build_pso(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* kernel_name
) {
    id<MTLFunction> fn = [library newFunctionWithName:kernel_name];
    if (!fn) {
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    (void)err;
    return pso;
}

static void write_device_name(id<MTLDevice> device, char* buf, size_t cap) {
    if (!buf || cap == 0) {
        return;
    }
    const char* utf8 = device.name.UTF8String;
    if (!utf8) {
        buf[0] = '\0';
        return;
    }
    size_t len = strlen(utf8);
    size_t copy_len = len < (cap - 1) ? len : (cap - 1);
    if (copy_len > 0) {
        memcpy(buf, utf8, copy_len);
    }
    buf[copy_len] = '\0';
}

static id<MTLBuffer> make_shared_buffer_copy(
    id<MTLDevice> device,
    const void* ptr,
    size_t byte_len
) {
    size_t alloc_len = byte_len < 4 ? 4 : byte_len;
    id<MTLBuffer> buf = [device newBufferWithLength:alloc_len options:MTLResourceStorageModeShared];
    if (!buf) {
        return nil;
    }
    if (byte_len > 0 && ptr != NULL) {
        memcpy(buf.contents, ptr, byte_len);
    } else {
        memset(buf.contents, 0, alloc_len);
    }
    return buf;
}

static id<MTLBuffer> make_shared_buffer_zero(id<MTLDevice> device, size_t byte_len) {
    size_t alloc_len = byte_len < 4 ? 4 : byte_len;
    id<MTLBuffer> buf = [device newBufferWithLength:alloc_len options:MTLResourceStorageModeShared];
    if (!buf) {
        return nil;
    }
    memset(buf.contents, 0, alloc_len);
    return buf;
}

static uint64_t compute_elapsed_ns(id<MTLCommandBuffer> cmd, uint64_t t0, uint64_t t1) {
    double gpu_start = cmd.GPUStartTime;
    double gpu_end = cmd.GPUEndTime;
    if (gpu_end > gpu_start && gpu_start > 0.0) {
        return (uint64_t)((gpu_end - gpu_start) * 1000000000.0);
    }
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    if (info.denom == 0 || t1 < t0) {
        return 0;
    }
    uint64_t delta = t1 - t0;
    __uint128_t ns = ((__uint128_t)delta * (__uint128_t)info.numer) / (__uint128_t)info.denom;
    return (uint64_t)ns;
}

int mq_metal_init(
    const uint8_t* metallib_bytes,
    size_t metallib_len,
    char* device_name_buf,
    size_t name_cap
) {
    @autoreleasepool {
        pthread_mutex_lock(&g_metal_mutex);
        if (g_initialized) {
            write_device_name(g_device, device_name_buf, name_cap);
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_OK;
        }

        if (!metallib_bytes || metallib_len == 0) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_INVALID_ARG;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_NO_DEVICE;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_NO_DEVICE;
        }

        dispatch_data_t data = dispatch_data_create(
            metallib_bytes,
            metallib_len,
            NULL,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT
        );
        if (!data) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_LIBRARY_COMPILE;
        }

        NSError* err = nil;
        id<MTLLibrary> library = [device newLibraryWithData:data error:&err];
        (void)err;
        if (!library) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_LIBRARY_COMPILE;
        }

        id<MTLComputePipelineState> pso_ternary = build_pso(device, library, @"ternary_2bit_gemv_i8");
        id<MTLComputePipelineState> pso_bitcos = build_pso(device, library, @"bitcos_gemv_i8");
        id<MTLComputePipelineState> pso_tequila = build_pso(device, library, @"tequila_deadzone_ste_q16");
        id<MTLComputePipelineState> pso_sinkhorn = build_pso(device, library, @"mhc_sinkhorn_4x4_q16");
        id<MTLComputePipelineState> pso_engram = build_pso(device, library, @"engram_ngram_gather_q16");

        if (!pso_ternary || !pso_bitcos || !pso_tequila || !pso_sinkhorn || !pso_engram) {
            pthread_mutex_unlock(&g_metal_mutex);
            return MQ_ERR_PIPELINE_CREATE;
        }

        g_device = device;
        g_queue = queue;
        g_library = library;
        g_pso_ternary = pso_ternary;
        g_pso_bitcos = pso_bitcos;
        g_pso_tequila = pso_tequila;
        g_pso_sinkhorn = pso_sinkhorn;
        g_pso_engram = pso_engram;
        g_initialized = YES;

        write_device_name(g_device, device_name_buf, name_cap);
        pthread_mutex_unlock(&g_metal_mutex);
        return MQ_OK;
    }
}

int mq_metal_ternary_2bit_gemv(
    const uint8_t* w_packed,
    size_t w_bytes,
    const int8_t* x,
    uint32_t rows,
    uint32_t cols,
    int32_t* y_out,
    uint64_t* gpu_ns_out
) {
    @autoreleasepool {
        if (!g_initialized) {
            return MQ_ERR_NOT_INITIALIZED;
        }
        if (rows == 0) {
            if (gpu_ns_out) {
                *gpu_ns_out = 0;
            }
            return MQ_OK;
        }
        if (!y_out) {
            return MQ_ERR_INVALID_ARG;
        }
        size_t cols_ceil4 = ((size_t)cols + 3u) / 4u;
        size_t expected_w_bytes = (size_t)rows * cols_ceil4;
        if (w_bytes < expected_w_bytes) {
            return MQ_ERR_INVALID_ARG;
        }
        if (cols > 0 && (!w_packed || !x)) {
            return MQ_ERR_INVALID_ARG;
        }

        id<MTLDevice> device = g_device;
        id<MTLCommandQueue> queue = g_queue;
        id<MTLComputePipelineState> pso = g_pso_ternary;

        id<MTLBuffer> w_buf = make_shared_buffer_copy(device, w_packed, w_bytes);
        id<MTLBuffer> x_buf = make_shared_buffer_copy(device, x, (size_t)cols);
        size_t y_bytes = (size_t)rows * sizeof(int32_t);
        id<MTLBuffer> y_buf = make_shared_buffer_zero(device, y_bytes);
        if (!w_buf || !x_buf || !y_buf) {
            return MQ_ERR_BUFFER_ALLOC;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return MQ_ERR_CMD_ENCODE;
        }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) {
            return MQ_ERR_CMD_ENCODE;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:y_buf offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&cols length:sizeof(uint32_t) atIndex:4];

        NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tg_threads = max_tg < 64 ? max_tg : 64;
        if (tg_threads == 0) {
            tg_threads = 1;
        }
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize tg = MTLSizeMake(tg_threads, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        uint64_t t0 = mach_continuous_time();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = mach_continuous_time();

        if (cmd.status != MTLCommandBufferStatusCompleted || cmd.error != nil) {
            return MQ_ERR_EXECUTION;
        }

        memcpy(y_out, y_buf.contents, y_bytes);
        if (gpu_ns_out) {
            *gpu_ns_out = compute_elapsed_ns(cmd, t0, t1);
        }
        return MQ_OK;
    }
}

int mq_metal_bitcos_gemv(
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
    @autoreleasepool {
        if (!g_initialized) {
            return MQ_ERR_NOT_INITIALIZED;
        }
        if (rows == 0) {
            if (gpu_ns_out) {
                *gpu_ns_out = 0;
            }
            return MQ_OK;
        }
        if (!y_out || !row_sign_byte_offsets) {
            return MQ_ERR_INVALID_ARG;
        }
        size_t p_row_bytes = ((size_t)cols + 7u) / 8u;
        size_t expected_p_bytes = (size_t)rows * p_row_bytes;
        if (p_bytes < expected_p_bytes) {
            return MQ_ERR_INVALID_ARG;
        }
        if (cols > 0 && (!presence || !x)) {
            return MQ_ERR_INVALID_ARG;
        }

        id<MTLDevice> device = g_device;
        id<MTLCommandQueue> queue = g_queue;
        id<MTLComputePipelineState> pso = g_pso_bitcos;

        id<MTLBuffer> p_buf = make_shared_buffer_copy(device, presence, p_bytes);
        id<MTLBuffer> s_buf = make_shared_buffer_copy(device, signs, s_bytes);
        size_t offsets_bytes = (size_t)rows * sizeof(uint32_t);
        id<MTLBuffer> off_buf = make_shared_buffer_copy(device, row_sign_byte_offsets, offsets_bytes);
        id<MTLBuffer> x_buf = make_shared_buffer_copy(device, x, (size_t)cols);
        size_t y_bytes = (size_t)rows * sizeof(int32_t);
        id<MTLBuffer> y_buf = make_shared_buffer_zero(device, y_bytes);
        if (!p_buf || !s_buf || !off_buf || !x_buf || !y_buf) {
            return MQ_ERR_BUFFER_ALLOC;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return MQ_ERR_CMD_ENCODE;
        }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) {
            return MQ_ERR_CMD_ENCODE;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:p_buf offset:0 atIndex:0];
        [enc setBuffer:s_buf offset:0 atIndex:1];
        [enc setBuffer:off_buf offset:0 atIndex:2];
        [enc setBuffer:x_buf offset:0 atIndex:3];
        [enc setBuffer:y_buf offset:0 atIndex:4];
        [enc setBytes:&rows length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&cols length:sizeof(uint32_t) atIndex:6];

        NSUInteger simd_w = pso.threadExecutionWidth;
        if (simd_w == 0) {
            simd_w = 32;
        }
        if (simd_w > pso.maxTotalThreadsPerThreadgroup) {
            simd_w = pso.maxTotalThreadsPerThreadgroup;
        }
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize tg = MTLSizeMake(simd_w, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        uint64_t t0 = mach_continuous_time();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = mach_continuous_time();

        if (cmd.status != MTLCommandBufferStatusCompleted || cmd.error != nil) {
            return MQ_ERR_EXECUTION;
        }

        memcpy(y_out, y_buf.contents, y_bytes);
        if (gpu_ns_out) {
            *gpu_ns_out = compute_elapsed_ns(cmd, t0, t1);
        }
        return MQ_OK;
    }
}

int mq_metal_tequila_ste_q16(
    int32_t* weights_q16,
    const int32_t* grads_q16,
    uint32_t len,
    int32_t delta_q16,
    int32_t lr_q16,
    int32_t lambda_q16,
    uint64_t* gpu_ns_out
) {
    @autoreleasepool {
        if (!g_initialized) {
            return MQ_ERR_NOT_INITIALIZED;
        }
        if (len == 0) {
            if (gpu_ns_out) {
                *gpu_ns_out = 0;
            }
            return MQ_OK;
        }
        if (!weights_q16 || !grads_q16) {
            return MQ_ERR_INVALID_ARG;
        }

        id<MTLDevice> device = g_device;
        id<MTLCommandQueue> queue = g_queue;
        id<MTLComputePipelineState> pso = g_pso_tequila;

        size_t vec_bytes = (size_t)len * sizeof(int32_t);
        id<MTLBuffer> w_buf = make_shared_buffer_copy(device, weights_q16, vec_bytes);
        id<MTLBuffer> g_buf = make_shared_buffer_copy(device, grads_q16, vec_bytes);
        if (!w_buf || !g_buf) {
            return MQ_ERR_BUFFER_ALLOC;
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            return MQ_ERR_CMD_ENCODE;
        }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) {
            return MQ_ERR_CMD_ENCODE;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:0 atIndex:0];
        [enc setBuffer:g_buf offset:0 atIndex:1];
        [enc setBytes:&len length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&delta_q16 length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&lr_q16 length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&lambda_q16 length:sizeof(int32_t) atIndex:5];

        NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tg_threads = max_tg < 256 ? max_tg : 256;
        if (tg_threads == 0) {
            tg_threads = 1;
        }
        NSUInteger num_tgs = ((NSUInteger)len + tg_threads - 1) / tg_threads;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
        [enc endEncoding];

        uint64_t t0 = mach_continuous_time();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = mach_continuous_time();

        if (cmd.status != MTLCommandBufferStatusCompleted || cmd.error != nil) {
            return MQ_ERR_EXECUTION;
        }

        memcpy(weights_q16, w_buf.contents, vec_bytes);
        if (gpu_ns_out) {
            *gpu_ns_out = compute_elapsed_ns(cmd, t0, t1);
        }
        return MQ_OK;
    }
}
