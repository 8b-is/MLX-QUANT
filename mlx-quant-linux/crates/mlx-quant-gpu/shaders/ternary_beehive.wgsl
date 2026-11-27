// Pure-integer WebGPU compute shaders for MLX-QUANT BeeHive ternary & BITCOS GEMV.
// Zero floating-point operations.

struct GemvParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> x_i8: array<u32>;
@group(0) @binding(2) var<storage, read_write> y_out: array<i32>;
@group(0) @binding(3) var<storage, read> w_packed: array<u32>;
@group(0) @binding(4) var<storage, read> presence: array<u32>;
@group(0) @binding(5) var<storage, read> signs: array<u32>;
@group(0) @binding(6) var<storage, read> row_sign_byte_offsets: array<u32>;

fn read_x_i8(col: u32) -> i32 {
    let word = x_i8[col >> 2u];
    let raw = i32((word >> ((col & 3u) * 8u)) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}

fn read_w_u8(byte_idx: u32) -> u32 {
    let word = w_packed[byte_idx >> 2u];
    return (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

fn read_presence_u8(byte_idx: u32) -> u32 {
    let word = presence[byte_idx >> 2u];
    return (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

fn read_signs_u8(byte_idx: u32) -> u32 {
    let word = signs[byte_idx >> 2u];
    return (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

// 2-bit LSB-first ternary GEMV: {0, 1, 2} -> {-1, 0, +1}, code 3 -> 0.
// Row stride is cols_ceil4 = (cols + 3) / 4 bytes per row.
@compute @workgroup_size(64, 1, 1)
fn ternary_2bit_gemv_i8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let cols = params.cols;
    let cols_ceil4 = (cols + 3u) >> 2u;
    let row_base = row * cols_ceil4;
    var acc: i32 = 0;

    for (var b: u32 = 0u; b < cols_ceil4; b = b + 1u) {
        let packed = read_w_u8(row_base + b);
        let col_base = b << 2u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let col = col_base + k;
            if (col < cols) {
                let code = (packed >> (k * 2u)) & 0x03u;
                let xv = read_x_i8(col);
                if (code == 2u) {
                    acc = acc + xv;
                } else if (code == 0u) {
                    acc = acc - xv;
                }
            }
        }
    }

    y_out[row] = acc;
}

// BITCOS sparse ternary GEMV: presence bitmap (ceil(cols/8) bytes/row) + compacted signs.
@compute @workgroup_size(64, 1, 1)
fn bitcos_gemv_i8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let cols = params.cols;
    let p_bytes = (cols + 7u) >> 3u;
    let p_base = row * p_bytes;
    let s_base = row_sign_byte_offsets[row];
    let rem_bits = cols & 7u;

    var acc: i32 = 0;
    var sign_idx: u32 = 0u;

    for (var b: u32 = 0u; b < p_bytes; b = b + 1u) {
        var p_byte = read_presence_u8(p_base + b);
        if (b + 1u == p_bytes && rem_bits != 0u) {
            p_byte = p_byte & ((1u << rem_bits) - 1u);
        }
        let col_base = b << 3u;
        while (p_byte != 0u) {
            let bit = firstTrailingBit(p_byte);
            p_byte = p_byte & (p_byte - 1u);
            let col = col_base + bit;
            let xv = read_x_i8(col);
            let s_byte = read_signs_u8(s_base + (sign_idx >> 3u));
            let s_bit = (s_byte >> (sign_idx & 7u)) & 1u;
            sign_idx = sign_idx + 1u;
            if (s_bit != 0u) {
                acc = acc + xv;
            } else {
                acc = acc - xv;
            }
        }
    }

    y_out[row] = acc;
}
