#![cfg_attr(not(feature = "std"), no_std)]
//! GPU compute backends and bare-metal command-ring builders for `mlx-quant`.
//!
//! - [`pm4`]: `#![no_std]` 64-byte aligned AMD CDNA3 Type-3 PM4 packet ring builder.
//! - [`metal`]: Safe Apple Silicon Metal GPU runtime bridge (`MetalRuntime`).
//! - Embedded compute shader sources for Metal, WebGPU (WGSL), and CUDA.

pub mod metal;
pub mod pm4;

pub use metal::{GpuError, MetalRuntime};
pub use pm4::*;

/// Embedded Metal Shading Language source for MLX-QUANT BeeHive kernels.
pub const METAL_SOURCE_TERNARY_BEEHIVE: &str = include_str!("../shaders/mlx_quant_beehive.metal");
/// Embedded WebGPU WGSL compute shader source for ternary & BITCOS GEMV.
pub const WGSL_TERNARY_BEEHIVE: &str = include_str!("../shaders/ternary_beehive.wgsl");
/// Embedded CUDA C++ compute kernel source for ternary & BITCOS GEMV.
pub const CUDA_TERNARY_BEEHIVE: &str = include_str!("../shaders/ternary_beehive.cu");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pm4_ring_alignment_wraparound_doorbell_and_validation() {
        let mut ring: Pm4Ring<16> = Pm4Ring::new();
        assert_eq!(core::mem::align_of::<Pm4Ring<16>>(), 64);
        assert_eq!(core::mem::align_of::<Pm4Packet64>(), 64);
        assert_eq!(core::mem::size_of::<Pm4Packet64>(), 64);
        assert_eq!((&ring as *const Pm4Ring<16> as usize) % 64, 0);
        assert_eq!(ring.capacity_words(), 16);
        assert_eq!(ring.free_words(), 16);
        assert_eq!(ring.doorbell_tail(), 0);

        // Invalid counts and headers must be rejected cleanly without mutating doorbell_tail.
        assert_eq!(
            ring.submit_type3(Pm4Opcode::Nop, &[]),
            Err(Pm4Error::InvalidCount)
        );
        let oversized = [0u32; 16];
        assert_eq!(
            ring.submit_type3(Pm4Opcode::Nop, &oversized),
            Err(Pm4Error::InvalidCount)
        );
        assert_eq!(
            ring.submit_raw_type3(&[0x0000_1000, 0x1234]),
            Err(Pm4Error::InvalidHeader)
        );
        let bad_hdr = pm4_type3_header(Pm4Opcode::Nop, 3).expect("valid header");
        assert_eq!(
            ring.submit_raw_type3(&[bad_hdr, 1, 2]),
            Err(Pm4Error::InvalidCount)
        );
        assert_eq!(ring.doorbell_tail(), 0);

        // Push 12 words (DispatchDirect = 5 words + AcquireMem = 8 words? 5 + 7 = 12)
        let t1 = ring
            .push_dispatch_direct(256, 1, 1, 0x0000_0001)
            .expect("dispatch_direct fits");
        assert_eq!(t1, 5);
        assert_eq!(ring.used_words(), 5);
        assert_eq!(ring.doorbell_tail(), 5);

        let t2 = ring
            .push_nop(6)
            .expect("7-word nop fits");
        assert_eq!(t2, 12);
        assert_eq!(ring.used_words(), 12);
        assert_eq!(ring.free_words(), 4);

        // Next 8-word packet won't fit until consumer drains some space.
        assert_eq!(
            ring.push_acquire_mem(0x0000_0003, 4096, 0x1000_0000),
            Err(Pm4Error::RingFull)
        );
        assert_eq!(ring.consume(13), Err(Pm4Error::ConsumeUnderflow));

        // Consume first 12 words so write wraps around slot 12..16 -> 0..4!
        ring.consume(12).expect("consume 12 words");
        assert_eq!(ring.used_words(), 0);
        assert_eq!(ring.free_words(), 16);

        let t3 = ring
            .push_acquire_mem(0x0000_000F, 0x1_0000_2000, 0xDEAD_BEEF_CAFE_0000)
            .expect("8-word acquire_mem wraps cleanly across ring end");
        assert_eq!(t3, 20);
        assert_eq!(ring.wptr(), 20);
        assert_eq!(ring.rptr(), 12);
        assert_eq!(ring.used_words(), 8);
        assert_eq!(ring.doorbell_tail(), 20);
        assert_eq!(ring.packets_submitted(), 3);

        // Verify wrapped slots at indices 12..15 and 0..3 match Pm4Packet64::acquire_mem
        let expected_pkt = Pm4Packet64::acquire_mem(0x0000_000F, 0x1_0000_2000, 0xDEAD_BEEF_CAFE_0000);
        let expected_slice = expected_pkt.active_slice().expect("active slice");
        for (idx, &word) in expected_slice.iter().enumerate() {
            assert_eq!(ring.peek_slot(12 + idx), word);
        }

        // Verify FNV-1a digest changes when ring mutates and is deterministic.
        let d_before = ring.fnv1a_ring_digest();
        let d_active_before = ring.fnv1a_active_digest();
        assert_ne!(d_before, 0);
        assert_ne!(d_active_before, 0);

        let mut ring2: Pm4Ring<16> = Pm4Ring::new();
        ring2
            .push_acquire_mem(0x0000_000F, 0x1_0000_2000, 0xDEAD_BEEF_CAFE_0000)
            .expect("submit identical packet at offset 0");
        assert_eq!(ring2.fnv1a_active_digest(), d_active_before);

        ring.push_write_data(0x8000_1000, &[0xAA55_0001, 0xBB66_0002])
            .expect("write_data fits");
        assert_ne!(ring.fnv1a_ring_digest(), d_before);
        assert_ne!(ring.fnv1a_active_digest(), d_active_before);
    }

    #[test]
    fn embedded_shader_sources_are_present_and_non_empty() {
        assert!(METAL_SOURCE_TERNARY_BEEHIVE.contains("ternary_2bit_gemv_i8"));
        assert!(METAL_SOURCE_TERNARY_BEEHIVE.contains("bitcos_gemv_i8"));
        assert!(METAL_SOURCE_TERNARY_BEEHIVE.contains("tequila_deadzone_ste_q16"));
        assert!(METAL_SOURCE_TERNARY_BEEHIVE.contains("mhc_sinkhorn_4x4_q16"));
        assert!(METAL_SOURCE_TERNARY_BEEHIVE.contains("engram_ngram_gather_q16"));
        assert!(WGSL_TERNARY_BEEHIVE.contains("ternary_2bit_gemv_i8"));
        assert!(WGSL_TERNARY_BEEHIVE.contains("bitcos_gemv_i8"));
        assert!(CUDA_TERNARY_BEEHIVE.contains("ternary_2bit_gemv_i8"));
        assert!(CUDA_TERNARY_BEEHIVE.contains("bitcos_gemv_i8"));
    }

    #[cfg(has_metal_bridge)]
    #[test]
    fn live_metal_device_smoke_and_edge_cases() {
        let rt = MetalRuntime::new().expect("MetalRuntime::new must succeed on macOS");
        let name = rt.device_name();
        assert!(!name.is_empty(), "Metal device name should not be empty");
        std::println!("Live Metal device: {name}");

        // Zero-row / zero-len edge cases must succeed cleanly without Metal buffer errors.
        let mut empty_y: [i32; 0] = [];
        assert_eq!(rt.ternary_2bit_gemv(&[], &[], &mut empty_y, 0, 0), Ok(0));
        assert_eq!(
            rt.bitcos_gemv(&[], &[], &[], &[], &mut empty_y, 0, 0),
            Ok(0)
        );
        let mut empty_w: [i32; 0] = [];
        assert_eq!(rt.tequila_ste_q16(&mut empty_w, &[], 100, 10, 10), Ok(0));

        // Zero-column, non-zero row case exercises 4-byte minimum Metal buffer allocation.
        let mut zero_col_y = [999i32; 4];
        let ns = rt
            .ternary_2bit_gemv(&[], &[], &mut zero_col_y, 4, 0)
            .expect("zero-col ternary GEMV succeeds");
        let _ = ns;
        assert_eq!(zero_col_y, [0, 0, 0, 0]);

        // Mismatched buffer bounds return InvalidBufferLength before FFI call.
        assert_eq!(
            rt.ternary_2bit_gemv(&[0u8; 1], &[0i8; 8], &mut zero_col_y, 2, 8),
            Err(GpuError::InvalidBufferLength)
        );
        let mut w_short = [0i32; 3];
        assert_eq!(
            rt.tequila_ste_q16(&mut w_short, &[0i32; 4], 16384, 655, 66),
            Err(GpuError::InvalidBufferLength)
        );
    }

    #[cfg(has_metal_bridge)]
    fn run_ternary_2bit_parity_case(rt: &MetalRuntime, rows: usize, cols: usize, seed_init: u64) {
        use mlx_quant_core::ratio::splitmix64;
        use mlx_quant_core::ternary::{Trit, pack_mlx_ternary_row, ternary_2bit_gemv_scalar};

        let stride = cols.div_ceil(4);
        let mut w_packed = vec![0u8; rows * stride];
        let mut row_trits = vec![Trit::Zero; cols];
        let mut seed = seed_init;

        for r in 0..rows {
            for t in &mut row_trits {
                seed = splitmix64(seed);
                *t = match seed % 3 {
                    0 => Trit::Neg,
                    1 => Trit::Zero,
                    _ => Trit::Pos,
                };
            }
            pack_mlx_ternary_row(&row_trits, &mut w_packed[r * stride..(r + 1) * stride]);
        }
        // Inject raw 0b11 code into row 0 byte 0 bits [1:0] (must map to 0 on both CPU and GPU).
        if rows > 0 && stride > 0 {
            w_packed[0] |= 0b11;
        }

        let mut x = vec![0i8; cols];
        for (c, slot) in x.iter_mut().enumerate() {
            seed = splitmix64(seed ^ (c as u64));
            *slot = (seed & 0xFF) as i8;
        }
        if cols >= 4 {
            x[0] = i8::MIN;
            x[1] = i8::MAX;
            x[cols - 2] = i8::MIN;
            x[cols - 1] = i8::MAX;
        }

        let mut cpu_y = vec![0i32; rows];
        let mut gpu_y = vec![0i32; rows];
        ternary_2bit_gemv_scalar(&w_packed, &x, &mut cpu_y, rows, cols);
        let gpu_ns = rt
            .ternary_2bit_gemv(&w_packed, &x, &mut gpu_y, rows, cols)
            .expect("Metal ternary_2bit_gemv succeeds");
        std::println!(
            "Metal {} ternary_2bit_gemv {rows}x{cols}: gpu_ns={gpu_ns}",
            rt.device_name()
        );
        assert_eq!(gpu_y, cpu_y, "ternary_2bit_gemv parity mismatch on {rows}x{cols}");
    }

    #[cfg(has_metal_bridge)]
    fn run_bitcos_parity_case(rt: &MetalRuntime, rows: usize, cols: usize, seed_init: u64) {
        use mlx_quant_core::ratio::splitmix64;
        use mlx_quant_core::ternary::{Trit, bitcos_gemv_scalar, bitcos_pack_into};

        let p_stride = cols.div_ceil(8);
        let mut presence = vec![0u8; rows * p_stride];
        let mut signs = Vec::new();
        let mut row_sign_byte_offsets = vec![0u32; rows];
        let mut row_trits = vec![Trit::Zero; cols];
        let mut seed = seed_init;

        for r in 0..rows {
            row_sign_byte_offsets[r] = signs.len() as u32;
            if r == rows / 2 {
                // Deliberately keep one full all-zero row (nnz == 0).
                row_trits.fill(Trit::Zero);
            } else {
                for t in &mut row_trits {
                    seed = splitmix64(seed);
                    *t = match seed & 7 {
                        0 | 1 => Trit::Pos,
                        2 | 3 => Trit::Neg,
                        _ => Trit::Zero,
                    };
                }
            }
            let max_s_bytes = cols.div_ceil(8);
            let mut row_signs = vec![0u8; max_s_bytes];
            let (_nnz, used_s_bytes) = bitcos_pack_into(
                &row_trits,
                &mut presence[r * p_stride..(r + 1) * p_stride],
                &mut row_signs,
            );
            signs.extend_from_slice(&row_signs[..used_s_bytes]);
        }

        let mut x = vec![0i8; cols];
        for (c, slot) in x.iter_mut().enumerate() {
            seed = splitmix64(seed ^ ((c as u64) << 7));
            *slot = (seed & 0xFF) as i8;
        }
        if cols >= 4 {
            x[0] = i8::MIN;
            x[1] = i8::MAX;
            x[cols - 2] = i8::MIN;
            x[cols - 1] = i8::MAX;
        }

        let mut cpu_y = vec![0i32; rows];
        let mut gpu_y = vec![0i32; rows];
        bitcos_gemv_scalar(
            &presence,
            &signs,
            &row_sign_byte_offsets,
            &x,
            &mut cpu_y,
            rows,
            cols,
        );
        let gpu_ns = rt
            .bitcos_gemv(
                &presence,
                &signs,
                &row_sign_byte_offsets,
                &x,
                &mut gpu_y,
                rows,
                cols,
            )
            .expect("Metal bitcos_gemv succeeds");
        std::println!(
            "Metal {} bitcos_gemv {rows}x{cols} (sign_bytes={}): gpu_ns={gpu_ns}",
            rt.device_name(),
            signs.len()
        );
        assert_eq!(gpu_y, cpu_y, "bitcos_gemv parity mismatch on {rows}x{cols}");
    }

    #[cfg(has_metal_bridge)]
    #[test]
    fn live_metal_ternary_2bit_gemv_matches_core_oracle_256x512_and_19x73() {
        let rt = MetalRuntime::new().expect("MetalRuntime::new");
        run_ternary_2bit_parity_case(&rt, 256, 512, 0x1111_2222_3333_4444);
        run_ternary_2bit_parity_case(&rt, 19, 73, 0xAAAA_BBBB_CCCC_DDDD);
    }

    #[cfg(has_metal_bridge)]
    #[test]
    fn live_metal_bitcos_gemv_matches_core_oracle_256x512_and_19x73() {
        let rt = MetalRuntime::new().expect("MetalRuntime::new");
        run_bitcos_parity_case(&rt, 256, 512, 0x5555_6666_7777_8888);
        run_bitcos_parity_case(&rt, 19, 73, 0x9999_0000_1234_5678);
    }

    #[cfg(has_metal_bridge)]
    #[test]
    fn live_metal_tequila_ste_q16_matches_core_oracle_256x512_and_19x73() {
        use mlx_quant_core::beehive::{TequilaConfig, tequila_ste_step_q16};
        use mlx_quant_core::ratio::{Q16_ONE, splitmix64};

        let rt = MetalRuntime::new().expect("MetalRuntime::new");
        let cfg = TequilaConfig {
            delta_q16: Q16_ONE >> 2,
            lr_q16: 655,
            lambda_q16: 66,
        };

        for (rows, cols) in [(256usize, 512usize), (19usize, 73usize)] {
            let len = rows * cols;
            let mut weights = vec![0i32; len];
            let mut grads = vec![0i32; len];
            let mut seed = 0xCAFE_BABE_DEAD_BEEFu64 ^ (len as u64);

            for i in 0..len {
                seed = splitmix64(seed);
                // Half inside deadzone [-delta, +delta], half outside, plus signed gradients.
                let span = if i & 1 == 0 {
                    cfg.delta_q16 * 2
                } else {
                    Q16_ONE * 8
                };
                weights[i] = ((seed as i64 % (span as i64 * 2 + 1)) - span as i64) as i32;
                seed = splitmix64(seed);
                grads[i] = ((seed as i64 % (Q16_ONE as i64 * 8 + 1)) - (Q16_ONE as i64 * 4)) as i32;
            }

            // Inject explicit boundary & extreme saturating cases.
            if len >= 10 {
                weights[0] = 0;
                grads[0] = 123_456;
                weights[1] = cfg.delta_q16 - 1;
                grads[1] = -98_765;
                weights[2] = cfg.delta_q16;
                grads[2] = 98_765;
                weights[3] = -cfg.delta_q16 + 1;
                grads[3] = -65_536;
                weights[4] = -cfg.delta_q16;
                grads[4] = 65_536;
                weights[5] = i32::MIN;
                grads[5] = i32::MAX;
                weights[6] = i32::MAX;
                grads[6] = i32::MIN;
                weights[7] = -1;
                grads[7] = 1;
                weights[8] = 1;
                grads[8] = -1;
                weights[9] = i32::MIN;
                grads[9] = i32::MIN;
            }

            let mut cpu_weights = weights.clone();
            let mut gpu_weights = weights;

            tequila_ste_step_q16(&mut cpu_weights, &grads, cfg);
            let gpu_ns = rt
                .tequila_ste_with_config(&mut gpu_weights, &grads, cfg)
                .expect("Metal tequila_ste_q16 succeeds");
            std::println!(
                "Metal {} tequila_ste_q16 {rows}x{cols} ({len} elems): gpu_ns={gpu_ns}",
                rt.device_name()
            );
            assert_eq!(
                gpu_weights, cpu_weights,
                "tequila_ste_q16 bit-for-bit parity mismatch on {rows}x{cols}"
            );
        }

        // Also test negative delta_q16 and negative lambda_q16 with extreme saturation.
        let neg_cfg = TequilaConfig {
            delta_q16: -1024,
            lr_q16: 65_536,
            lambda_q16: -4096,
        };
        let mut cpu_neg = vec![0i32, -500, 500, i32::MIN, i32::MAX, -100_000, 100_000];
        let grads_neg = vec![10_000, i32::MAX, i32::MIN, i32::MAX, i32::MIN, 54_321, -54_321];
        let mut gpu_neg = cpu_neg.clone();
        tequila_ste_step_q16(&mut cpu_neg, &grads_neg, neg_cfg);
        rt.tequila_ste_with_config(&mut gpu_neg, &grads_neg, neg_cfg)
            .expect("Metal negative-cfg tequila_ste_q16 succeeds");
        assert_eq!(gpu_neg, cpu_neg);
    }
}
