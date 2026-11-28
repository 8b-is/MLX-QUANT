use super::*;

#[test]
fn ratio_arithmetic_gcd_asm_and_q16_ops() {
    assert_eq!(Ratio::new(6, 4), BONOBO_RESONANCE);
    assert_eq!(Ratio::new(9, 3), ASYMMETRIC_CRITICAL_WEIGHT);
    assert_eq!(Ratio::new(0, 5), Ratio::ZERO);
    assert_eq!(Ratio::new(5, 0), Ratio::ZERO);
    assert_eq!(Ratio::new(7, 7), Ratio::ONE);

    assert_eq!(Ratio::new(1, 3).add_exact(Ratio::new(1, 6)), Ratio::new(1, 2));
    assert_eq!(Ratio::new(3, 4).sub_exact(Ratio::new(1, 2)), Ratio::new(1, 4));
    assert_eq!(Ratio::new(1, 4).sub_exact(Ratio::new(1, 2)), Ratio::ZERO);
    assert_eq!(BONOBO_RESONANCE.mul_exact(Ratio::new(2, 3)), Ratio::ONE);
    assert_eq!(BONOBO_RESONANCE.div_exact(Ratio::new(3, 4)), Ratio::new(2, 1));
    assert_eq!(BONOBO_RESONANCE.div_exact(Ratio::ZERO), Ratio::ZERO);

    assert!(Ratio::new(2, 3) < Ratio::new(3, 4));
    assert_eq!(Ratio::new(3, 2).to_q16_u64(), 98_304);
    assert_eq!(Ratio::new(3, 2).to_decimal_parts_x100(), (1, 50));
    assert_eq!(format!("{}", Ratio::new(6, 4)), "3/2");

    let large = Ratio::from_u128((u64::MAX as u128) * 6, (u64::MAX as u128) * 4);
    assert_eq!(large, BONOBO_RESONANCE);
    let huge = Ratio::from_u128((1u128 << 100) + 1, (1u128 << 99) + 1);
    assert!(huge.q >= 1);

    let (holds, surplus) = Ratio::new(12, 5).ultra_zeno_witness(11, -1);
    assert!(holds && surplus > 0);

    assert_eq!(q16_mul(Q16_ONE, Q16_ONE / 2), Q16_ONE / 2);
    assert_eq!(q16_mul(i32::MAX, Q16_ONE * 2), i32::MAX);
    assert_eq!(q16_div(Q16_ONE * 3, Q16_ONE * 2), 98_304);
    assert_eq!(q16_div(Q16_ONE, 0), 0);
    assert_eq!(q16_div(i32::MIN, -1), i32::MAX);

    let edge_gcd = [
        (0u64, 0u64),
        (0, 42),
        (42, 0),
        (1, u64::MAX),
        (u64::MAX, u64::MAX),
        (1 << 63, 1 << 32),
        (1 << 63, (1 << 63) - 1),
        (48 * 1_000_003, 18 * 1_000_003),
    ];
    for &(a, b) in &edge_gcd {
        let g = gcd_binary_u64(a, b);
        assert_eq!(gcd_binary_u128(a as u128, b as u128), g as u128);
        assert_eq!(Ratio::new_fast(a, b), Ratio::new(a, b));
        #[cfg(target_arch = "aarch64")]
        assert_eq!(gcd_binary_u64_asm(a, b), g);
    }

    let mut seed = 0xDEAD_BEEF_CAFE_BABEu64;
    for i in 0..10_000u64 {
        seed = splitmix64(seed ^ i);
        let a = seed;
        seed = splitmix64(seed);
        let b = seed;
        let g = gcd_binary_u64(a, b);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(gcd_binary_u64_asm(a, b), g);
        assert_eq!(Ratio::new_fast(a, b), Ratio::new(a, b));
    }
}

#[test]
fn epoch_bump_arena_slots_and_8_thread_zero_overlap_stress() {
    const ARENA_BYTES: usize = 256 * 1024;
    let mut arena = Box::new(EpochBumpArena::<ARENA_BYTES>::new());
    assert_eq!(arena.capacity_bytes(), ARENA_BYTES);
    assert_eq!(arena.epoch(), 0);
    assert_eq!(arena.used_bytes(), 0);

    // Exhaust all 64 slot bits and verify release semantics.
    let mut slots = [0u8; 64];
    for slot_ref in &mut slots {
        *slot_ref = arena.claim_slot().expect("slot available");
    }
    assert!(arena.claim_slot().is_none());
    assert_eq!(arena.active_slot_mask(), u64::MAX);
    assert!(!arena.release_slot(64));
    for &s in &slots {
        assert!(arena.release_slot(s));
        assert!(!arena.release_slot(s));
    }
    assert_eq!(arena.active_slot_mask(), 0);

    // Non-power-of-two alignments above 64 must be rejected.
    assert!(arena.alloc_bytes(64, 96).is_none());

    for epoch_round in 0..3u64 {
        assert_eq!(arena.epoch(), epoch_round);
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let arena_ref: &EpochBumpArena<ARENA_BYTES> = &arena;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(8);
            for tid in 0..8usize {
                handles.push(scope.spawn(move || {
                    let slot = arena_ref.claim_slot().expect("8 < 64 slots");
                    let mut local_ranges = Vec::new();

                    for iter in 0..24usize {
                        let size = ((tid * 17 + iter * 31) % 257) + 1;
                        let align = if iter % 3 == 0 { 128 } else { 64 };
                        if let Some(buf) = arena_ref.alloc_bytes(size, align) {
                            let start = buf.as_ptr() as usize;
                            assert_eq!(start % align.max(64), 0);
                            let tag = ((tid as u8) << 4) ^ (iter as u8);
                            buf.fill(tag);
                            assert!(buf.iter().all(|&b| b == tag));
                            local_ranges.push((start, start + size));
                        }
                        if let Some(words) = arena_ref.alloc_slice_mut::<i32>(8) {
                            let start = words.as_ptr() as usize;
                            assert_eq!(start % 64, 0);
                            assert!(words.iter().all(|&v| v == 0));
                            words.fill((tid as i32) * 1000 + (iter as i32));
                            local_ranges.push((start, start + core::mem::size_of_val(words)));
                        }
                    }

                    assert!(arena_ref.release_slot(slot));
                    local_ranges
                }));
            }
            for h in handles {
                ranges.extend(h.join().expect("worker thread panicked"));
            }
        });

        assert_eq!(arena.active_slot_mask(), 0);
        assert!(arena.used_bytes() > 0);
        assert!(arena.high_water_bytes() >= arena.used_bytes());
        assert!(arena.alloc_count() >= ranges.len());

        ranges.sort_unstable_by_key(|r| r.0);
        for pair in ranges.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "overlapping allocations detected: {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }

        let prev_hwm = arena.high_water_bytes();
        arena.reset_epoch();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.epoch(), epoch_round + 1);
        assert!(arena.high_water_bytes() >= prev_hwm);
    }

    // OOM boundary check.
    assert!(arena.alloc_bytes(ARENA_BYTES + 1, 64).is_none());
}

#[test]
fn epoch_bump_arena_slice_handles_unaligned_and_heap_buffers() {
    // Deliberately shift by +3 bytes to verify 64-byte alignment helper.
    let mut backing = vec![0xAAu8; 4 * 1024 * 1024 + 128];
    let mut slice_arena = EpochBumpArenaSlice::new(&mut backing[3..]);
    assert!(slice_arena.capacity_bytes() >= 4 * 1024 * 1024);

    let mut ranges = Vec::new();
    {
        let arena_ref = &slice_arena;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(8);
            for tid in 0..8usize {
                handles.push(scope.spawn(move || {
                    let mut local = Vec::new();
                    for step in 0..32usize {
                        let sz = 512 + (step * 64);
                        let s = arena_ref
                            .alloc_bytes(sz, 128)
                            .expect("4 MiB slice arena has ample room");
                        let addr = s.as_ptr() as usize;
                        assert_eq!(addr % 128, 0);
                        s.fill(tid as u8);
                        local.push((addr, addr + sz));
                    }
                    local
                }));
            }
            for h in handles {
                ranges.extend(h.join().expect("slice worker panicked"));
            }
        });
    }
    ranges.sort_unstable_by_key(|r| r.0);
    for pair in ranges.windows(2) {
        assert!(pair[0].1 <= pair[1].0);
    }

    slice_arena.reset_epoch();
    assert_eq!(slice_arena.used_bytes(), 0);
    assert_eq!(slice_arena.epoch(), 1);
}

#[test]
fn ternary_encodings_bitcos_spqt_int4_and_hadamard_parity() {
    // 1. Exhaustive 2-bit OmniLLL and MLX-QUANT roundtrips across all 256 bytes.
    for b in 0u8..=255 {
        let omni = unpack_trits_4(b);
        let repacked_omni = pack_trits_4(omni);
        assert_eq!(unpack_trits_4(repacked_omni), omni);
        let mlx = unpack_mlx_ternary_4(b);
        let repacked_mlx = pack_mlx_ternary_4(mlx);
        assert_eq!(unpack_mlx_ternary_4(repacked_mlx), mlx);
    }
    assert_eq!(Trit::Neg.encode_mlx_2bit(), 0);
    assert_eq!(Trit::Zero.encode_mlx_2bit(), 1);
    assert_eq!(Trit::Pos.encode_mlx_2bit(), 2);
    assert_eq!(Trit::decode_mlx_2bit(3), Trit::Zero);

    // 2. 64-lane + odd-length 2-bit dot & GEMV parity (including i8::MIN and code=3).
    let mut packed_64 = [0xFFu8; 16];
    let mut w_64 = [0i8; 64];
    let mut a_64 = [0i8; 64];
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    for (i, slot) in a_64.iter_mut().enumerate() {
        seed = splitmix64(seed ^ (i as u64));
        *slot = match i {
            0 => i8::MIN,
            1 => i8::MAX,
            _ => (seed & 0xFF) as i8,
        };
    }
    // Code 3 in every lane must yield 0.
    assert_eq!(ternary_2bit_dot_64_scalar(&packed_64, &a_64), 0);

    let mut trits_64 = [Trit::Zero; 64];
    for i in 0..64 {
        seed = splitmix64(seed);
        let t = match seed % 3 {
            0 => Trit::Neg,
            1 => Trit::Zero,
            _ => Trit::Pos,
        };
        trits_64[i] = t;
        w_64[i] = t.as_i8();
    }
    pack_mlx_ternary_row(&trits_64, &mut packed_64);
    assert_eq!(
        ternary_2bit_dot_64_scalar(&packed_64, &a_64),
        ternary_dot_64_scalar(&w_64, &a_64)
    );

    // Unaligned 19x73 GEMV parity: ternary_2bit_gemv_scalar vs bitcos_gemv_scalar vs naive i8 dot.
    let rows = 19usize;
    let cols = 73usize;
    let w_stride = cols.div_ceil(4);
    let p_stride = cols.div_ceil(8);
    let mut w_2bit = vec![0u8; rows * w_stride];
    let mut presence = vec![0u8; rows * p_stride];
    let mut signs = Vec::new();
    let mut row_sign_offsets = vec![0u32; rows];
    let mut w_dense = vec![0i8; rows * cols];
    let mut x = vec![0i8; cols];
    for (c, slot) in x.iter_mut().enumerate() {
        seed = splitmix64(seed ^ (c as u64));
        *slot = if c == 0 { i8::MIN } else { (seed & 0xFF) as i8 };
    }

    let mut total_nnz = 0usize;
    for r in 0..rows {
        let mut row_trits = vec![Trit::Zero; cols];
        for c in 0..cols {
            seed = splitmix64(seed);
            let t = if r == 7 || seed % 5 < 3 {
                Trit::Zero // Includes all-zero row 7 edge case.
            } else if seed & 1 == 0 {
                Trit::Pos
            } else {
                Trit::Neg
            };
            row_trits[c] = t;
            w_dense[r * cols + c] = t.as_i8();
        }
        pack_mlx_ternary_row(&row_trits, &mut w_2bit[r * w_stride..(r + 1) * w_stride]);
        row_sign_offsets[r] = signs.len() as u32;
        let row_nnz = row_trits.iter().filter(|&&t| t != Trit::Zero).count();
        let mut row_signs = vec![0u8; row_nnz.div_ceil(8)];
        let (packed_nnz, _) = bitcos_pack_into(
            &row_trits,
            &mut presence[r * p_stride..(r + 1) * p_stride],
            &mut row_signs,
        );
        assert_eq!(packed_nnz, row_nnz);
        total_nnz += row_nnz;
        signs.extend_from_slice(&row_signs);

        let mut unpacked = vec![Trit::Zero; cols];
        assert_eq!(
            bitcos_unpack_into(
                cols,
                &presence[r * p_stride..(r + 1) * p_stride],
                &row_signs,
                &mut unpacked
            ),
            row_nnz
        );
        assert_eq!(unpacked, row_trits);
    }

    let mut y_2bit = vec![0i32; rows];
    let mut y_bitcos = vec![0i32; rows];
    ternary_2bit_gemv_scalar(&w_2bit, &x, &mut y_2bit, rows, cols);
    bitcos_gemv_scalar(&presence, &signs, &row_sign_offsets, &x, &mut y_bitcos, rows, cols);
    for r in 0..rows {
        let expected = ternary_dot_scalar(&w_dense[r * cols..(r + 1) * cols], &x);
        assert_eq!(y_2bit[r], expected, "2bit row {r}");
        assert_eq!(y_bitcos[r], expected, "bitcos row {r}");
    }
    assert!(bitcos_effective_bpw(rows * cols, total_nnz) < Ratio::new(2, 1));

    // 3. SPQT Base-243 exhaustive roundtrip + sentinel rejection + dot parity.
    for b in 0u8..=242 {
        let t = unpack_spqt_5(b).expect("valid SPQT byte");
        assert_eq!(pack_spqt_5(t), b);
    }
    for s in 243u8..=255 {
        assert!(unpack_spqt_5(s).is_none());
    }
    let mut spqt_packed = pack_spqt_64(&w_64);
    assert_eq!(
        spqt_dot_64_scalar(&spqt_packed, &a_64),
        Some(ternary_dot_64_scalar(&w_64, &a_64))
    );
    assert_eq!(
        spqt_dot_scalar(&spqt_packed, &a_64, 64),
        Some(ternary_dot_64_scalar(&w_64, &a_64))
    );
    spqt_packed[4] = SPQT_CORRUPT_SENTINEL;
    assert!(unpack_spqt_64(&spqt_packed).is_none());
    assert!(spqt_dot_64_scalar(&spqt_packed, &a_64).is_none());
    assert!(spqt_dot_scalar(&spqt_packed, &a_64, 64).is_none());

    // 4. INT4 packed & GEMM parity (odd k=37).
    for b in 0u8..=255 {
        let (lo, hi) = int4_unpack_byte(b);
        assert!((-8..=7).contains(&lo));
        assert!((-8..=7).contains(&hi));
        assert_eq!(int4_pack_byte(lo, hi), b);
    }
    let m = 5usize;
    let n = 7usize;
    let k = 37usize;
    let k_bytes = k.div_ceil(2);
    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; n * k];
    let mut a_p = vec![0u8; m * k_bytes];
    let mut b_p = vec![0u8; n * k_bytes];
    for r in 0..m {
        for c in 0..k {
            seed = splitmix64(seed);
            a_i8[r * k + c] = ((seed % 15) as i8) - 7;
        }
        assert_eq!(
            int4_pack_symmetric(&a_i8[r * k..(r + 1) * k], &mut a_p[r * k_bytes..(r + 1) * k_bytes]),
            k_bytes
        );
    }
    for r in 0..n {
        for c in 0..k {
            seed = splitmix64(seed);
            b_i8[r * k + c] = ((seed % 15) as i8) - 7;
        }
        assert_eq!(
            int4_pack_symmetric(&b_i8[r * k..(r + 1) * k], &mut b_p[r * k_bytes..(r + 1) * k_bytes]),
            k_bytes
        );
    }
    let mut c_out = vec![0i32; m * n];
    int4_gemm_scalar(&a_p, &b_p, &mut c_out, m, n, k);
    for r in 0..m {
        for col in 0..n {
            let expected: i32 = (0..k)
                .map(|t| (a_i8[r * k + t] as i32) * (b_i8[col * k + t] as i32))
                .sum();
            assert_eq!(c_out[r * n + col], expected);
        }
    }

    // 5. H-BitLinear WHT + INT4 + BitNet AbsMean.
    assert!(!hadamard_wht_inplace_i32(&mut []));
    assert!(!hadamard_wht_inplace_i32(&mut [1, 2, 3]));
    let orig = [12, -7, 3, 19, -25, 8, 0, -4];
    let mut wht = orig;
    assert!(hadamard_wht_inplace_i32(&mut wht));
    assert!(hadamard_wht_inplace_i32(&mut wht));
    for i in 0..8 {
        assert_eq!(wht[i], orig[i] * 8);
    }

    let mut spike = [0i32; 16];
    spike[0] = 16 * Q16_ONE;
    spike[1] = -2 * Q16_ONE;
    let peak_before = spike.iter().map(|v| v.abs()).max().unwrap_or(0);
    assert!(hadamard_wht_normalized_q16(&mut spike));
    let peak_after = spike.iter().map(|v| v.abs()).max().unwrap_or(0);
    assert!(peak_after < peak_before);

    let mut act_i4 = [0i8; 16];
    let scale = quantize_activations_int4_symmetric(&spike, &mut act_i4);
    assert!(scale > 0);
    assert!(act_i4.iter().all(|&q| (-7..=7).contains(&q)));
    assert!(act_i4.iter().any(|&q| q.abs() == 7));

    let w_q16 = [Q16_ONE, -Q16_ONE, Q16_ONE >> 2, -(Q16_ONE >> 2)];
    let mut q_trits = [Trit::Zero; 4];
    let gamma = quantize_weights_ternary_absmean_q16(&w_q16, &mut q_trits);
    assert_eq!(gamma, Ratio::new(5, 8));
    assert_eq!(q_trits, [Trit::Pos, Trit::Neg, Trit::Zero, Trit::Zero]);
}

#[test]
fn tensor_dtypes_shapes_and_checked_views() {
    assert_eq!(DType::Ternary2Bit.bytes_for_numel(65), 17);
    assert_eq!(DType::BitcosSparse.bytes_for_numel(65), 18);
    assert_eq!(DType::SpqtBase243.bytes_for_numel(64), 13);
    assert_eq!(DType::Int4Packed.bytes_for_numel(37), 19);
    assert_eq!(DType::Int8.bytes_for_numel(11), 11);
    assert_eq!(DType::Int32.bytes_for_numel(4), 16);
    assert_eq!(DType::Q15_16.bytes_for_numel(4), 16);
    assert_eq!(DType::Float16Raw.bytes_for_numel(5), 10);
    assert_eq!(DType::Q15_16.name(), "q15_16");

    assert!(Shape::from_slice(&[]).is_none());
    assert!(Shape::from_slice(&[1, 2, 3, 4, 5]).is_none());
    let s2 = Shape::new_2d(4, 16);
    assert_eq!(s2.numel(), 64);
    assert_eq!(s2.as_slice(), &[4, 16]);

    let mut raw = [0u8; 16];
    assert!(TensorView::new_checked(&raw[..15], s2, DType::Ternary2Bit).is_err());
    let tv = TensorView::new_checked(&raw, s2, DType::Ternary2Bit).expect("16 bytes fits 64 trits");
    assert_eq!(tv.numel(), 64);
    assert_eq!(tv.dtype(), DType::Ternary2Bit);
    assert_eq!(tv.shape(), s2);

    let mut tm = TensorMut::new_checked(&mut raw, s2, DType::Ternary2Bit).expect("mutable view");
    tm.data_mut()[0] = 0xAA;
    assert_eq!(tm.as_view().data()[0], 0xAA);

    // Sparse BITCOS tensor view only requires presence bitmap floor (8 bytes for 64 weights).
    assert!(TensorView::new_checked(&raw[..8], s2, DType::BitcosSparse).is_ok());
    assert!(TensorView::new_checked(&raw[..7], s2, DType::BitcosSparse).is_err());

    let bad_shape = Shape {
        dims: [1; 4],
        ndim: 0,
    };
    assert_eq!(
        TensorView::new_checked(&raw, bad_shape, DType::Int8),
        Err(TensorError::InvalidShape)
    );
}

#[test]
fn beehive_tequila_engram_sinkhorn_ramanujan_and_wta() {
    // 1. Tequila STE deadzone reactivation vs active zone.
    let mut weights = [Q16_ONE, -(Q16_ONE >> 3), Q16_ONE >> 4, -Q16_ONE];
    let grads = [Q16_ONE >> 1, 0, Q16_ONE >> 2, -(Q16_ONE >> 1)];
    let cfg = TequilaConfig {
        delta_q16: Q16_ONE >> 2,
        lr_q16: 6554,   // ~0.1
        lambda_q16: 6554, // ~0.1
    };
    let orig = weights;
    tequila_ste_step_q16(&mut weights, &grads, cfg);
    // Index 1 had grad == 0 inside deadzone (w < 0), so dynamic bias reactivated it away from 0!
    assert_ne!(weights[1], orig[1]);
    assert!(weights[1] > orig[1]); // w < 0 -> dyn_bias < 0 -> w - update > w
    // Index 0 had |w| >= delta_q16 -> standard STE step without lambda bias.
    let expected0 = orig[0] - (((grads[0] as i64 * cfg.lr_q16 as i64) >> 16) as i32);
    assert_eq!(weights[0], expected0);

    // 2. EngramTable insert, lookup, and gated residual merge.
    let mut engram = EngramTable::<64, 8>::new();
    let ngram = [35u32, 33, 91];
    let mem = [Q16_ONE; 8];
    for head in 0..4 {
        engram.insert_ngram(&ngram, head, &mem);
        assert_eq!(engram.lookup_ngram(&ngram, head), Some(&mem));
    }
    let mut hidden = [Q16_ONE >> 1; 8];
    let gate = engram.gated_residual_merge_q16(&mut hidden, &ngram, 4);
    assert!(gate > 0 && gate <= Q16_ONE);
    assert!(hidden.iter().all(|&v| v > (Q16_ONE >> 1)));

    // 3. mHC 4x4 Sinkhorn-Knopp Birkhoff doubly-stochastic convergence.
    let raw = [
        [4 * Q16_ONE, Q16_ONE, Q16_ONE >> 1, Q16_ONE >> 2],
        [Q16_ONE >> 1, 3 * Q16_ONE, Q16_ONE, Q16_ONE],
        [Q16_ONE, Q16_ONE >> 2, 5 * Q16_ONE, Q16_ONE],
        [0, Q16_ONE, Q16_ONE, 4 * Q16_ONE],
    ];
    let ds = mhc_sinkhorn_4x4_q16(raw, 20);
    for (r, row) in ds.iter().enumerate() {
        let row_sum: i32 = row.iter().sum();
        assert!(
            (row_sum - Q16_ONE).abs() <= 4,
            "row {r} sum {row_sum} did not converge to {Q16_ONE}"
        );
        assert!(row.iter().all(|&v| v >= 0));
    }
    for c in 0..4 {
        let col_sum: i32 = ds.iter().map(|row| row[c]).sum();
        assert!(
            (col_sum - Q16_ONE).abs() <= 4,
            "col {c} sum {col_sum} did not converge to {Q16_ONE}"
        );
    }

    // 4. Ramanujan bipartite expander regular row degrees & distinct columns.
    let out_dim = 32usize;
    let in_dim = 64usize;
    let deg = 6usize;
    let mut edges = vec![(0usize, 0usize); out_dim * deg];
    let written = ramanujan_bipartite_edges(out_dim, in_dim, deg, 42, &mut edges);
    assert_eq!(written, out_dim * deg);
    let mut col_deg = vec![0usize; in_dim];
    for r in 0..out_dim {
        let row_edges = &edges[r * deg..(r + 1) * deg];
        for (idx, &(er, ec)) in row_edges.iter().enumerate() {
            assert_eq!(er, r);
            assert!(ec < in_dim);
            assert!(!row_edges[..idx].iter().any(|&(_, prev)| prev == ec));
            col_deg[ec] += 1;
        }
    }
    assert!(col_deg.iter().all(|&d| d >= 1));

    // 5. Mushroom-Body WTA routing + novelty habituation shift + exact Q16_ONE sum.
    let scores = [100_000, 95_000, 10_000, -5_000];
    let mut hab = [0i32; 4];
    let r1 = mushroom_body_wta_route::<2>(&scores, Some(&mut hab), 4_000);
    assert_eq!(r1.active_count, 2);
    assert!(r1.quorum_met);
    assert_eq!(r1.indices[0], 0);
    assert_eq!(r1.weights_q16[..r1.active_count].iter().sum::<i32>(), Q16_ONE);

    // Repeat strong stimulus on unit 0 until habituation shifts winner to unit 1!
    for _ in 0..10 {
        let _ = mushroom_body_wta_route::<1>(&[100_000, 0, 0, 0], Some(&mut hab), 4_000);
    }
    let r_shifted = mushroom_body_wta_route::<2>(&scores, Some(&mut hab), 4_000);
    assert_eq!(r_shifted.indices[0], 1);
    assert_eq!(
        r_shifted.weights_q16[..r_shifted.active_count]
            .iter()
            .sum::<i32>(),
        Q16_ONE
    );
}

#[test]
fn anyrate_2plane_progressive_ternary_q16_slashes_error() {
    let mut seed = 0x0002_6113_4402_u64;
    let mut w = [0i32; 128];
    let mut x = [0i32; 128];
    for i in 0..128 {
        seed = splitmix64(seed);
        // Deterministic signed Q15.16 weights in roughly [-2.0, +2.0]
        w[i] = (seed as i32) >> 14;
        seed = splitmix64(seed);
        x[i] = (seed as i32) >> 15;
    }
    let mut p0 = [Trit::Zero; 128];
    let mut p1 = [Trit::Zero; 128];
    let mut scratch = [0i32; 128];
    let scales = quantize_anyrate_2plane_q16(&w, &mut p0, &mut p1, &mut scratch);
    assert!(scales.scale0_q16 > 0);
    assert!(scales.scale1_q16 > 0);
    assert!(scales.scale1_q16 < scales.scale0_q16);

    let mut sq_err_p1: i128 = 0;
    let mut sq_err_p2: i128 = 0;
    for i in 0..128 {
        let q1 = (p0[i].as_i8() as i128) * (scales.scale0_q16 as i128);
        let q2 = q1 + (p1[i].as_i8() as i128) * (scales.scale1_q16 as i128);
        let wi = w[i] as i128;
        sq_err_p1 += (wi - q1) * (wi - q1);
        sq_err_p2 += (wi - q2) * (wi - q2);
    }
    // Plane 2 cuts squared error by >3x over Plane 1 in pure integer Q15.16!
    assert!(
        sq_err_p2 * 3 < sq_err_p1,
        "expected >3x MSE drop: p1={sq_err_p1} vs p2={sq_err_p2}"
    );

    let exact_dot: i64 = (w
        .iter()
        .zip(&x)
        .map(|(&wi, &xi)| (wi as i128) * (xi as i128))
        .sum::<i128>()
        >> 16) as i64;
    let d1 = dot_anyrate_2plane_q16(&p0, &p1, scales, &x, 1);
    let d2 = dot_anyrate_2plane_q16(&p0, &p1, scales, &x, 2);
    assert!(
        (d2 - exact_dot).abs() < (d1 - exact_dot).abs(),
        "2-plane dot err {} should beat 1-plane dot err {}",
        (d2 - exact_dot).abs(),
        (d1 - exact_dot).abs()
    );
}
