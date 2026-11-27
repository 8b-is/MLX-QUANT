//! Comprehensive unit, parity, concurrency, and integration tests for `mlx-quant-linux`.

use crate::ExecutionMode;
use crate::agx_doorbell::AgxDoorbell;
use crate::allocator::BumpAllocator;
use crate::amd_mi300x::{AmdComputeRing, Pm4Error};
use crate::antigravity::Antigravity;
use crate::asteroid_belt::AsteroidBelt;
use crate::big_bang::BigBang;
use crate::cmb::CosmicMicrowaveBackground;
use crate::dark_energy::DarkEnergy;
use crate::dark_matter::DarkMatter;
use crate::event_horizon::EventHorizon;
use crate::fusion_polarity::FusionPolarity;
use crate::gravitational_lensing::GravitationalLensing;
use crate::hawking_radiation::HawkingRadiation;
use crate::io_blackhole::IoBlackhole;
use crate::magnetar::Magnetar;
use crate::marley_and_fifike::TheGuardians;
use crate::math::matmul::{ComputeSingularity, MatmulError};
use crate::multiverse::Multiverse;
use crate::polar_queue::PolarQueue;
use crate::pulsar::Pulsar;
use crate::quantum_entanglement::QuantumEntanglement;
use crate::quasar::Quasar;
use crate::redshift::Redshift;
use crate::supernova::Supernova;
use crate::tensor::{DType, Shape, Tensor};
use crate::time_dilation::TimeDilation;
use crate::white_hole::WhiteHole;
use crate::wormhole::Wormhole;

use mlx_quant_core::beehive::{
    EngramTable, TequilaConfig, mhc_sinkhorn_4x4_q16, mushroom_body_wta_route,
    ramanujan_bipartite_edges, tequila_ste_step_q16,
};
use mlx_quant_core::ratio::{BONOBO_RESONANCE, Q16_ONE, Ratio, splitmix64};
use mlx_quant_core::ternary::{
    Trit, bitcos_gemv_scalar, bitcos_pack_into, hadamard_wht_normalized_q16, int4_gemm_scalar,
    int4_pack_symmetric, pack_mlx_ternary_row, quantize_activations_int4_symmetric,
    quantize_weights_ternary_absmean_q16, ternary_2bit_gemv_scalar,
};
use mlx_quant_gpu::GpuError;
use std::io::Cursor;

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BumpAllocator>();
};

#[test]
fn bump_allocator_default_64mib_and_8_thread_zero_overlap_and_epoch_reset() {
    let default_alloc = BumpAllocator::new();
    assert_eq!(default_alloc.capacity_bytes(), 64 * 1024 * 1024);
    assert_eq!(default_alloc.used_bytes(), 0);
    drop(default_alloc);

    let mut alloc = BumpAllocator::with_capacity(512 * 1024);
    assert_eq!(alloc.capacity_bytes(), 512 * 1024);
    assert_eq!(alloc.epoch(), 0);
    assert_eq!(alloc.used_bytes(), 0);

    // Non-power-of-two alignments > 64 are rejected cleanly.
    assert!(alloc.alloc_bytes(64, 96).is_none());

    for round in 0..3u64 {
        assert_eq!(alloc.epoch(), round);
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let alloc_ref = &alloc;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(8);
            for tid in 0..8usize {
                handles.push(scope.spawn(move || {
                    let slot = alloc_ref.claim_slot().expect("slot free");
                    let mut local = Vec::new();
                    for iter in 0..24usize {
                        let size = ((tid * 19 + iter * 37) % 300) + 1;
                        let align = if iter % 3 == 0 { 128 } else { 64 };
                        let buf = alloc_ref.alloc_bytes(size, align).expect("alloc fits");
                        let start = buf.as_ptr() as usize;
                        assert_eq!(start % align.max(64), 0);
                        let tag = ((tid as u8) << 5) ^ (iter as u8);
                        buf.fill(tag);
                        assert!(buf.iter().all(|&b| b == tag));
                        local.push((start, start + size));
                    }
                    assert!(alloc_ref.release_slot(slot));
                    local
                }));
            }
            for h in handles {
                ranges.extend(h.join().expect("worker panic"));
            }
        });

        assert_eq!(alloc.active_slot_mask(), 0);
        assert!(alloc.high_water_bytes() >= alloc.used_bytes());
        assert!(alloc.alloc_count() >= ranges.len());

        ranges.sort_unstable_by_key(|r| r.0);
        for w in ranges.windows(2) {
            assert!(w[0].1 <= w[1].0, "overlapping slices: {:?} vs {:?}", w[0], w[1]);
        }

        let (reclaimed, new_epoch) = HawkingRadiation::evaporate_cold_cache(&mut alloc);
        assert!(reclaimed > 0);
        assert_eq!(new_epoch, round + 1);
        assert_eq!(alloc.used_bytes(), 0);
    }

    // OOM check.
    assert!(alloc.alloc_tensor_bytes(alloc.capacity_bytes() + 1).is_none());
}

#[test]
fn tensor_slot_drop_and_views() {
    let mut alloc = BumpAllocator::with_capacity(64 * 1024);
    assert_eq!(alloc.active_slot_mask(), 0);

    {
        let mut t = Tensor::new(Shape::new_1d(16), DType::Int32, &alloc).expect("tensor");
        assert!(t.slot().is_some());
        assert_ne!(alloc.active_slot_mask(), 0);
        assert!(t.as_view().is_ok());
        assert!(t.as_view_mut().is_ok());
        assert!(t.as_i8_slice().is_none());
        let ints = t.as_i32_slice_mut().expect("i32 mut");
        ints.fill(42);
        let d1 = t.fnv1a_digest();
        t.zero();
        let d2 = t.fnv1a_digest();
        assert_ne!(d1, d2);
    }
    assert_eq!(alloc.active_slot_mask(), 0);

    // Unaligned new_raw should reject i32 casting safely.
    let mut unaligned = [0u8; 32];
    let mut raw_t = Tensor::new_raw(&mut unaligned[1..17], Shape::new_1d(4), DType::Int32)
        .expect("raw tensor");
    if !(raw_t.as_bytes().as_ptr() as usize).is_multiple_of(4) {
        assert!(raw_t.as_i32_slice_mut().is_none());
    }

    let _ = HawkingRadiation::evaporate_cold_cache(&mut alloc);
}

#[test]
fn compute_singularity_ternary_int4_and_bitcos_parity() {
    let alloc = BumpAllocator::with_capacity(512 * 1024);

    // 1. Ternary2Bit x Int8 -> Int32 (odd 19x73 shape)
    let rows = 19usize;
    let cols = 73usize;
    let stride = cols.div_ceil(4);
    let mut w_t = Tensor::new(Shape::new_2d(rows, cols), DType::Ternary2Bit, &alloc).expect("w_t");
    let mut x_t = Tensor::new(Shape::new_1d(cols), DType::Int8, &alloc).expect("x_t");
    let mut c_t = Tensor::new(Shape::new_1d(rows), DType::Int32, &alloc).expect("c_t");

    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut trits = vec![Trit::Zero; cols];
    for r in 0..rows {
        for t in &mut trits {
            seed = splitmix64(seed);
            *t = match seed % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            };
        }
        pack_mlx_ternary_row(&trits, &mut w_t.as_bytes_mut()[r * stride..(r + 1) * stride]);
    }
    for xi in x_t.as_i8_slice_mut().expect("i8 mut").iter_mut() {
        seed = splitmix64(seed);
        *xi = (seed & 0xFF) as i8;
    }

    ComputeSingularity::execute_matmul(&w_t, &x_t, &mut c_t).expect("ternary matmul");
    let mut expected = vec![0i32; rows];
    ternary_2bit_gemv_scalar(
        w_t.as_bytes(),
        x_t.as_i8_slice().expect("i8"),
        &mut expected,
        rows,
        cols,
    );
    assert_eq!(c_t.as_i32_slice().expect("i32"), expected.as_slice());

    // Dimension mismatch should return an error, not panic.
    let mut bad_c = Tensor::new(Shape::new_1d(rows + 1), DType::Int32, &alloc).expect("bad_c");
    assert_eq!(
        ComputeSingularity::execute_matmul(&w_t, &x_t, &mut bad_c),
        Err(MatmulError::DimensionMismatch)
    );

    // 2. Int4Packed x Int4Packed -> Int32 (odd k=37)
    let (m, n, k) = (7usize, 9usize, 37usize);
    let k_bytes = k.div_ceil(2);
    let mut a_t = Tensor::new(Shape::new_2d(m, k), DType::Int4Packed, &alloc).expect("a_t");
    let mut b_t = Tensor::new(Shape::new_2d(n, k), DType::Int4Packed, &alloc).expect("b_t");
    let mut c4_t = Tensor::new(Shape::new_2d(m, n), DType::Int32, &alloc).expect("c4_t");

    let mut tmp = vec![0i8; k];
    for r in 0..m {
        for v in &mut tmp {
            seed = splitmix64(seed);
            *v = ((seed % 15) as i8) - 7;
        }
        int4_pack_symmetric(&tmp, &mut a_t.as_bytes_mut()[r * k_bytes..(r + 1) * k_bytes]);
    }
    for col in 0..n {
        for v in &mut tmp {
            seed = splitmix64(seed);
            *v = ((seed % 15) as i8) - 7;
        }
        int4_pack_symmetric(&tmp, &mut b_t.as_bytes_mut()[col * k_bytes..(col + 1) * k_bytes]);
    }

    ComputeSingularity::execute_matmul(&a_t, &b_t, &mut c4_t).expect("int4 matmul");
    let mut expected_c4 = vec![0i32; m * n];
    int4_gemm_scalar(a_t.as_bytes(), b_t.as_bytes(), &mut expected_c4, m, n, k);
    assert_eq!(c4_t.as_i32_slice().expect("c4"), expected_c4.as_slice());

    // 3. BITCOS GEMV parity
    let p_stride = cols.div_ceil(8);
    let mut presence = vec![0u8; rows * p_stride];
    let mut signs = Vec::new();
    let mut offsets = Vec::with_capacity(rows);
    for r in 0..rows {
        offsets.push(signs.len() as u32);
        for t in &mut trits {
            seed = splitmix64(seed);
            *t = match seed & 3 {
                0 => Trit::Pos,
                1 => Trit::Neg,
                _ => Trit::Zero,
            };
        }
        let mut s_row = vec![0u8; p_stride];
        let (_nnz, s_used) = bitcos_pack_into(
            &trits,
            &mut presence[r * p_stride..(r + 1) * p_stride],
            &mut s_row,
        );
        signs.extend_from_slice(&s_row[..s_used]);
    }
    let x_vec = x_t.as_i8_slice().expect("x_i8").to_vec();
    let mut y_bitcos = vec![0i32; rows];
    let mut y_bitcos_ref = vec![0i32; rows];
    ComputeSingularity::execute_bitcos_gemv(
        &presence,
        &signs,
        &offsets,
        &x_vec,
        &mut y_bitcos,
        rows,
        cols,
    )
    .expect("bitcos");
    bitcos_gemv_scalar(&presence, &signs, &offsets, &x_vec, &mut y_bitcos_ref, rows, cols);
    assert_eq!(y_bitcos, y_bitcos_ref);
}

#[test]
fn agx_doorbell_and_polar_queue_live_and_simulator_parity() {
    let alloc = BumpAllocator::with_capacity(512 * 1024);
    let live = AgxDoorbell::new();
    let sim = AgxDoorbell::new_simulator();
    assert_eq!(sim.backend_mode(), "Portable Ring Simulator");
    #[cfg(target_os = "macos")]
    assert_eq!(live.backend_mode(), "Apple Metal Live GPU");

    let rows = 19usize;
    let cols = 73usize;
    let stride = cols.div_ceil(4);
    let mut w_packed = vec![0u8; rows * stride];
    let mut trits = vec![Trit::Zero; cols];
    let mut seed = 0x5566_7788_99AA_BBCCu64;
    for r in 0..rows {
        for t in &mut trits {
            seed = splitmix64(seed);
            *t = match seed % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            };
        }
        pack_mlx_ternary_row(&trits, &mut w_packed[r * stride..(r + 1) * stride]);
    }
    let mut x = vec![0i8; cols];
    for xi in &mut x {
        seed = splitmix64(seed);
        *xi = (seed & 0xFF) as i8;
    }

    let mut y_live = vec![0i32; rows];
    let mut y_sim = vec![0i32; rows];
    let mut y_ref = vec![0i32; rows];
    ternary_2bit_gemv_scalar(&w_packed, &x, &mut y_ref, rows, cols);
    live.submit_ternary_2bit_gemv(&w_packed, &x, &mut y_live, rows, cols)
        .expect("live submit");
    sim.submit_ternary_2bit_gemv(&w_packed, &x, &mut y_sim, rows, cols)
        .expect("sim submit");
    assert_eq!(y_live, y_ref);
    assert_eq!(y_sim, y_ref);
    assert_eq!(live.ring_counter(), 1);
    assert_eq!(sim.ring_counter(), 1);

    // Invalid bounds return GpuError::InvalidBufferLength
    assert_eq!(
        sim.submit_ternary_2bit_gemv(&[0u8; 1], &x, &mut y_sim, rows, cols),
        Err(GpuError::InvalidBufferLength)
    );

    // PolarQueue end-to-end arm attachment + tensor feed + GPU dispatch
    let mut pq = PolarQueue::new(&alloc);
    let aw = pq.attach_arm("w_arm").expect("aw");
    let ax = pq.attach_arm("x_arm").expect("ax");
    let ay = pq.attach_arm("y_arm").expect("ay");
    for i in 3..8 {
        assert!(pq.attach_arm(&format!("extra_{i}")).is_some());
    }
    assert!(pq.attach_arm("overflow").is_none());

    let mut wt = Tensor::new(Shape::new_2d(rows, cols), DType::Ternary2Bit, &alloc).expect("wt");
    wt.as_bytes_mut().copy_from_slice(&w_packed);
    let mut xt = Tensor::new(Shape::new_1d(cols), DType::Int8, &alloc).expect("xt");
    xt.as_bytes_mut().copy_from_slice(
        // SAFETY: &[i8] -> &[u8] identical size/alignment 1
        unsafe { core::slice::from_raw_parts(x.as_ptr().cast::<u8>(), cols) },
    );
    let yt = Tensor::new(Shape::new_1d(rows), DType::Int32, &alloc).expect("yt");

    pq.feed_tensor(aw, wt).expect("feed wt");
    pq.feed_tensor(ax, xt).expect("feed xt");
    pq.feed_tensor(ay, yt).expect("feed yt");
    let receipt = pq.collapse_and_execute(7);
    assert_eq!(receipt.active_arms, 3);
    assert_eq!(receipt.ring_id, 7);

    pq.dispatch_ternary_gemv(aw, ax, ay).expect("polar gemv");
    let out_slice = pq
        .arm(ay)
        .and_then(|arm| arm.tensor())
        .and_then(|t| t.as_i32_slice())
        .expect("out slice");
    assert_eq!(out_slice, y_ref.as_slice());

    // Verify BITCOS and Tequila Q15.16 STE parity through both live Metal and simulator doorbells.
    let p_stride = cols.div_ceil(8);
    let mut presence = vec![0u8; rows * p_stride];
    let mut signs = Vec::new();
    let mut offsets = Vec::with_capacity(rows);
    for r in 0..rows {
        offsets.push(signs.len() as u32);
        for t in &mut trits {
            seed = splitmix64(seed);
            *t = match seed & 3 {
                0 => Trit::Pos,
                1 => Trit::Neg,
                _ => Trit::Zero,
            };
        }
        let mut s_row = vec![0u8; p_stride];
        let (_nnz, s_used) = bitcos_pack_into(
            &trits,
            &mut presence[r * p_stride..(r + 1) * p_stride],
            &mut s_row,
        );
        signs.extend_from_slice(&s_row[..s_used]);
    }
    let mut b_live = vec![0i32; rows];
    let mut b_sim = vec![0i32; rows];
    let mut b_ref = vec![0i32; rows];
    bitcos_gemv_scalar(&presence, &signs, &offsets, &x, &mut b_ref, rows, cols);
    live.submit_bitcos_gemv(&presence, &signs, &offsets, &x, &mut b_live, rows, cols)
        .expect("live bitcos");
    sim.submit_bitcos_gemv(&presence, &signs, &offsets, &x, &mut b_sim, rows, cols)
        .expect("sim bitcos");
    assert_eq!(b_live, b_ref);
    assert_eq!(b_sim, b_ref);

    let cfg = TequilaConfig {
        delta_q16: Q16_ONE >> 2,
        lr_q16: 6_554,
        lambda_q16: 6_554,
    };
    let mut w_ref = [Q16_ONE, -(Q16_ONE >> 3), Q16_ONE >> 4, -Q16_ONE];
    let mut w_live = w_ref;
    let mut w_sim = w_ref;
    let grads = [Q16_ONE >> 1, 0, Q16_ONE >> 2, -(Q16_ONE >> 1)];
    tequila_ste_step_q16(&mut w_ref, &grads, cfg);
    live.submit_tequila_ste_q16(&mut w_live, &grads, cfg)
        .expect("live tequila");
    sim.submit_tequila_ste_q16(&mut w_sim, &grads, cfg)
        .expect("sim tequila");
    assert_eq!(w_live, w_ref);
    assert_eq!(w_sim, w_ref);
}

#[test]
fn io_blackhole_redshift_and_fusion_polarity_end_to_end() {
    let alloc = BumpAllocator::with_capacity(256 * 1024);
    let bh = IoBlackhole::new(&alloc);

    // Short read zero-padding verification.
    let src = [0xAAu8, 0xBB, 0xCC, 0xDD, 0xEE];
    let mut cursor = Cursor::new(&src[..]);
    let (tensor, bytes_read, digest) = bh
        .load_bytes_from_reader(&mut cursor, Shape::new_1d(16), DType::Int8)
        .expect("bounded read");
    assert_eq!(bytes_read, 5);
    assert_eq!(&tensor.as_bytes()[..5], &src);
    assert!(tensor.as_bytes()[5..16].iter().all(|&b| b == 0));
    assert_ne!(digest, 0);

    // Temp file read verification.
    let temp_path = std::env::temp_dir().join(format!("mlx_quant_bh_test_{}.bin", std::process::id()));
    std::fs::write(&temp_path, b"MLXQUANT123").expect("write temp");
    let mut file = std::fs::File::open(&temp_path).expect("open temp");
    let (ftensor, fread, _) = bh
        .suck_weights_from_disk(&mut file, Shape::new_1d(16), DType::Int8)
        .expect("file read");
    let _ = std::fs::remove_file(&temp_path);
    assert_eq!(fread, 11);
    assert_eq!(&ftensor.as_bytes()[..11], b"MLXQUANT123");
    assert_eq!(&ftensor.as_bytes()[11..16], &[0u8; 5]);

    // Redshift quantizers parity against mlx_quant_core::ternary oracles.
    let w_q16 = [Q16_ONE, -Q16_ONE, Q16_ONE >> 2, -(Q16_ONE >> 2), Q16_ONE * 2];
    let (w_tern, gamma) = Redshift::redshift_q16_to_ternary_2bit(&w_q16, &alloc);
    let mut ref_trits = [Trit::Zero; 5];
    let ref_gamma = quantize_weights_ternary_absmean_q16(&w_q16, &mut ref_trits);
    let mut ref_packed = [0u8; 2];
    pack_mlx_ternary_row(&ref_trits, &mut ref_packed);
    assert_eq!(gamma, ref_gamma);
    assert_eq!(w_tern.as_bytes(), &ref_packed);

    let x_q16 = [16 * Q16_ONE, -8 * Q16_ONE, 3 * Q16_ONE, 0, -Q16_ONE];
    let (x_int4, scale) = Redshift::redshift_q16_to_int4(&x_q16, &alloc);
    let mut ref_i4 = [0i8; 5];
    let ref_scale = quantize_activations_int4_symmetric(&x_q16, &mut ref_i4);
    let mut ref_i4_packed = [0u8; 3];
    int4_pack_symmetric(&ref_i4, &mut ref_i4_packed);
    assert_eq!(scale, ref_scale);
    assert_eq!(x_int4.as_bytes(), &ref_i4_packed);

    let bitcos = Redshift::redshift_trits_to_bitcos(&ref_trits, &alloc);
    assert_eq!(bitcos.nnz, 3);
    assert_eq!(bitcos.effective_bpw, Ratio::new(8, 5));

    // FusionPolarity H-BitLinear forward parity against step-by-step core calls.
    let cols = 16usize;
    let rows = 4usize;
    let mut w_mat_q16 = [0i32; 64];
    let mut s = 0x4444_5555_6666_7777u64;
    for slot in &mut w_mat_q16 {
        s = splitmix64(s);
        *slot = ((s as i64 % (Q16_ONE as i64 * 4)) - (Q16_ONE as i64 * 2)) as i32;
    }
    let (w_mat_tern, _) =
        Redshift::try_redshift_q16_to_ternary_2bit(&w_mat_q16, Shape::new_2d(rows, cols), &alloc)
            .expect("2d ternary");
    let mut act_in = [0i32; 16];
    for (i, slot) in act_in.iter_mut().enumerate() {
        *slot = ((i as i32) - 7) * Q16_ONE;
    }
    let mut act_fused = act_in;
    let (fused_out, fused_res) =
        FusionPolarity::h_bitlinear_forward_tensor(&w_mat_tern, &mut act_fused, &alloc)
            .expect("h_bitlinear_forward_tensor");

    let mut act_ref = act_in;
    assert!(hadamard_wht_normalized_q16(&mut act_ref));
    let mut q_ref = [0i8; 16];
    let scale_ref = quantize_activations_int4_symmetric(&act_ref, &mut q_ref);
    let mut y_ref = [0i32; 4];
    ternary_2bit_gemv_scalar(w_mat_tern.as_bytes(), &q_ref, &mut y_ref, rows, cols);
    assert_eq!(fused_res.absmax_scale_q16, scale_ref);
    assert_eq!(fused_out.as_i32_slice().expect("i32"), &y_ref);

    // 2D odd-dimension [19, 73] Redshift -> ComputeSingularity GEMV check.
    let (odd_r, odd_c) = (19usize, 73usize);
    let mut odd_w_q16 = vec![0i32; odd_r * odd_c];
    for slot in &mut odd_w_q16 {
        s = splitmix64(s);
        *slot = ((s as i64 % (Q16_ONE as i64 * 4)) - (Q16_ONE as i64 * 2)) as i32;
    }
    let (odd_w_t, _) =
        Redshift::try_redshift_q16_to_ternary_2bit(&odd_w_q16, Shape::new_2d(odd_r, odd_c), &alloc)
            .expect("odd 2d ternary");
    let mut odd_x = Tensor::new(Shape::new_1d(odd_c), DType::Int8, &alloc).expect("odd_x");
    for (idx, xi) in odd_x.as_i8_slice_mut().expect("i8").iter_mut().enumerate() {
        *xi = ((idx % 15) as i8) - 7;
    }
    let mut odd_y = Tensor::new(Shape::new_1d(odd_r), DType::Int32, &alloc).expect("odd_y");
    ComputeSingularity::execute_matmul(&odd_w_t, &odd_x, &mut odd_y).expect("odd matmul");
    let mut odd_y_ref = vec![0i32; odd_r];
    ternary_2bit_gemv_scalar(
        odd_w_t.as_bytes(),
        odd_x.as_i8_slice().expect("i8"),
        &mut odd_y_ref,
        odd_r,
        odd_c,
    );
    assert_eq!(odd_y.as_i32_slice().expect("i32"), odd_y_ref.as_slice());
}

#[test]
fn cosmology_modules_and_pm4_ring_behavior() {
    let alloc = BumpAllocator::with_capacity(64 * 1024);
    let (base, cap) = alloc.backing_addr_range();
    let horizon = EventHorizon::erect_firewall(base, cap);
    assert!(horizon.contains(base, cap));
    assert!(!horizon.contains(base.saturating_sub(1), cap));
    assert!(!horizon.contains(usize::MAX - 8, 16));

    // Antigravity L1 prefetch lines
    let buf = [1u8; 192];
    assert_eq!(Antigravity::levitate_slice(&buf), 3);
    assert_eq!(Antigravity::levitate_tensor(core::ptr::null(), 128), 0);

    // Pulsar timing
    assert_eq!(Pulsar::elapsed_ns_with_freq(100, 124, 24_000_000), 1_000);
    assert_eq!(Pulsar::elapsed_ns_with_freq(200, 100, 24_000_000), 0);
    let beacon = Pulsar::emit_sync_beacon();
    #[cfg(target_arch = "aarch64")]
    assert!(beacon.freq_hz > 0 && beacon.mode == ExecutionMode::Measured);

    // DarkEnergy Ratio arithmetic
    let scaled = DarkEnergy::accelerate_entropy(BONOBO_RESONANCE, Ratio::new(4, 3));
    assert_eq!(scaled, Ratio::new(2, 1));
    assert_eq!(
        DarkEnergy::dampen_entropy(scaled, Ratio::new(2, 1)),
        Ratio::ONE
    );

    // CMB circular KV ring buffer wraparound
    let mut cmb = CosmicMicrowaveBackground::<3>::new();
    for pos in 0..5u32 {
        cmb.push((pos as i32 + 1) * Q16_ONE, Q16_ONE, pos);
    }
    assert_eq!(cmb.len(), 3);
    assert_eq!(cmb.total_written(), 5);
    assert_eq!(cmb.get_oldest(0).expect("oldest").token_pos, 2);
    assert_eq!(cmb.get_latest(0).expect("latest").token_pos, 4);
    assert_eq!(cmb.dot_keys_q16(Q16_ONE), (3 + 4 + 5) * (Q16_ONE as i64));

    // GravitationalLensing speculative draft & verification
    let draft = GravitationalLensing::speculate_timelines(99, 5);
    assert_eq!(draft.len, 5);
    let mut target = draft.tokens[..5].to_vec();
    target[3] ^= 1;
    let receipt = GravitationalLensing::verify_speculative_tokens(draft.as_slice(), &target);
    assert_eq!(receipt.accepted_prefix, 3);
    assert_eq!(receipt.bonus_token, Some(target[3]));
    assert_eq!(receipt.acceptance_ratio, Ratio::new(3, 5));

    // Remaining cosmology descriptors & guardians
    let bb = BigBang::ignite_tensor_graph(&alloc, 24);
    assert_eq!(bb.mode, ExecutionMode::Measured);
    let t = Tensor::new(Shape::new_1d(16), DType::Int8, &alloc).expect("t");
    assert!(!Magnetar::lock_memory(&t).locked);
    assert!(!Magnetar::pin_to_core(2).pinned);
    assert_eq!(
        DarkMatter::ghost_map_sparse_tensor(65536, 16384).deduplication_ratio,
        Ratio::new(4, 1)
    );
    let (sg, sg_digest) = AsteroidBelt::gather_slices(&[b"abc", b"defg"]);
    assert_eq!(sg.total_bytes, 7);
    assert_ne!(sg_digest, 0);
    assert_eq!(Wormhole::open_p2p_tunnel(0x1000, 0x3000, 4096).bar_distance, 0x2000);
    assert_eq!(QuantumEntanglement::entangle_tensors(&t, "10.0.0.2").remote_str(), "10.0.0.2");
    assert_eq!(Quasar::emit_rdma_jet(&t, "10.0.0.3").target_str(), "10.0.0.3");
    assert!(!Supernova::default_safe_policy().enabled);
    assert!(!Supernova::trigger_burst(120_000, 90_000).enabled);

    let sched = TimeDilation::plan_micro_batches(10, 4);
    assert_eq!(sched.num_batches, 3);
    assert_eq!(sched.remainder, 2);
    let mut batch_count = 0usize;
    TimeDilation::micro_batch_loop(&[1, 2, 3, 4, 5], 2, |_, _| batch_count += 1);
    assert_eq!(batch_count, 3);

    let mut wh = WhiteHole::open_ejection_port(4);
    assert!(wh.emit_step(10, 2));
    assert!(wh.emit_step(2, 2));
    assert!(!wh.emit_step(11, 2));
    assert!(wh.status().halted_on_eos);

    let mut trace = Multiverse::join_swarm("NODE_0", 0, 123);
    let d_before = trace.state_digest;
    Multiverse::deposit_pheromone(&mut trace, 1, 456);
    assert_ne!(trace.state_digest, d_before);
    assert_eq!(trace.swarm_name(), "NODE_0");

    let guardian = TheGuardians::verify_integrity(&alloc, &horizon);
    assert!(guardian.healthy);
    assert_eq!(TheGuardians::fifike_universal_heal(), (25, 150));
    assert!(TheGuardians::marley_human_protect(&horizon));

    // AmdComputeRing PM4 checks
    let mut pm4 = AmdComputeRing::new();
    assert_eq!(pm4.mmio_execution_mode(), ExecutionMode::DesignedDescriptor);
    assert_eq!(pm4.consume(1), Err(Pm4Error::ConsumeUnderflow));
    pm4.acquire_barrier(0x3, 4096, 0x1000).expect("barrier");
    pm4.dispatch_kernel(8, 1, 1, 1).expect("dispatch");
    assert!(pm4.doorbell_tail() > 0);
    assert_ne!(pm4.fnv1a_active_digest(), 0);
    let drained = pm4.drain();
    assert_eq!(drained, 13);
    assert_eq!(pm4.used_words(), 0);

    // BeeHive core sanity
    let mut w = [Q16_ONE >> 3; 4];
    let g = [0; 4];
    let teq_cfg = TequilaConfig {
        delta_q16: Q16_ONE >> 2,
        lr_q16: 6_554,
        lambda_q16: 6_554,
    };
    tequila_ste_step_q16(&mut w, &g, teq_cfg);
    assert_ne!(w[0], Q16_ONE >> 3);
    let mut eng = EngramTable::<16, 4>::new();
    eng.insert_ngram(&[1, 2], 0, &[Q16_ONE; 4]);
    let mut hid = [Q16_ONE; 4];
    assert!(eng.gated_residual_merge_q16(&mut hid, &[1, 2], 1) > 0);
    let ds = mhc_sinkhorn_4x4_q16([[Q16_ONE; 4]; 4], 8);
    assert_eq!(ds[0].iter().sum::<i32>(), Q16_ONE);
    let mut edges = [(0usize, 0usize); 16];
    assert_eq!(ramanujan_bipartite_edges(4, 8, 4, 1, &mut edges), 16);
    let route = mushroom_body_wta_route::<2>(&[Q16_ONE, Q16_ONE >> 1], None, 0);
    assert_eq!(route.weights_q16[..2].iter().sum::<i32>(), Q16_ONE);

    // Offline A6000 telemetry parser check (zero float)
    let sample = r#"{
      "node_ip": "49.13.101.247",
      "gpu": {
        "name": "NVIDIA RTX 6000 Ada Generation",
        "vcpu_count": 48,
        "vram_used_mib": 19062,
        "vram_total_mib": 49140,
        "vram_free_mib": 30078,
        "vram_ratio_pq": "353/910",
        "gpu_util_pct": 0,
        "temp_celsius": 30,
        "power_milliwatts": 20160,
        "load_trit": 1
      },
      "healthy_service_ratio_pq": "3/4",
      "subsystem_f_improver": {
        "total_cycles": 47129,
        "mean_speedup_pq": "3758/2269"
      },
      "vlan_160tb": {
        "raid160_free_gib": 102312,
        "git160_free_gib": 14325,
        "total_synced_github_repos": 367
      }
    }"#;
    let snap = crate::a6000_telemetry::parse_a6000_status_json(sample).expect("a6000 json");
    assert_eq!(snap.gpu_name, "NVIDIA RTX 6000 Ada Generation");
    assert_eq!(snap.vram_free_mib, 30078);
    assert_eq!(snap.vram_ratio, Ratio::new(353, 910));
    assert_eq!(snap.improver_speedup, Ratio::new(3758, 2269));
    assert_eq!(snap.healthy_services, Ratio::new(3, 4));
    assert_eq!(snap.total_synced_github_repos, 367);
}
