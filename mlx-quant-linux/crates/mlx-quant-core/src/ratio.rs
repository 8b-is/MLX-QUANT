//! Exact canonical coprime rationals `p/q`, Stein binary GCD (with AArch64 inline ASM),
//! Q15.16 dyadic fixed-point arithmetic, and SplitMix64 deterministic PRNG.
//!
//! Zero floating-point operations anywhere in core state coordinates, accumulators,
//! or graph invariants.

use core::cmp::Ordering;
use core::fmt;

/// Dyadic shift for Q15.16 signed fixed-point (`2^16 = 65_536`).
pub const Q16_SHIFT: u32 = 16;

/// Unity in Q15.16 fixed-point (`65_536`).
pub const Q16_ONE: i32 = 1 << Q16_SHIFT;

/// Exact reduced rational number `p / q` with `q >= 1` and `gcd(p, q) == 1`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub p: u64,
    pub q: u64,
}

/// Rung-0 Bonobo target resonance `3/2` (+1 trit anchor).
pub const BONOBO_RESONANCE: Ratio = Ratio { p: 3, q: 2 };

/// Peter's Asymmetric Loss Modulation critical-token multiplier `3/1`.
pub const ASYMMETRIC_CRITICAL_WEIGHT: Ratio = Ratio { p: 3, q: 1 };

impl Default for Ratio {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

/// Stein's binary GCD on `u64` using hardware `trailing_zeros()` and shift-subtract.
#[inline]
pub const fn gcd_binary_u64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// Pure AArch64 inline-assembly Stein's binary GCD on `u64`.
///
/// Uses `orr` + `cbz` fallthrough for zero checks, `rbit` + `clz` + `lsr` initial
/// odd normalization, and a branchless `subs`/`csel`/`cneg` inner loop.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn gcd_binary_u64_asm(a: u64, b: u64) -> u64 {
    let out: u64;
    // SAFETY: Pure register-only Stein binary GCD on `u64` values. No memory or stack
    // is accessed, and all modified scratch/output registers are declared explicitly.
    unsafe {
        core::arch::asm!(
            "orr {out}, {a}, {b}",
            "cbz {a}, 4f",
            "cbz {b}, 4f",
            "rbit {tmp}, {a}",
            "rbit {tz}, {b}",
            "clz {tmp}, {tmp}",
            "clz {tz}, {tz}",
            "lsr {a}, {a}, {tmp}",
            "lsr {b}, {b}, {tz}",
            "cmp {tmp}, {tz}",
            "csel {shift}, {tmp}, {tz}, ls",
            "3:",
            "subs {tmp}, {b}, {a}",
            "csel {a}, {a}, {b}, cs",
            "cneg {b}, {tmp}, cc",
            "rbit {tz}, {tmp}",
            "clz {tz}, {tz}",
            "lsr {b}, {b}, {tz}",
            "b.ne 3b",
            "lsl {out}, {a}, {shift}",
            "4:",
            a = inout(reg) a => _,
            b = inout(reg) b => _,
            out = out(reg) out,
            shift = out(reg) _,
            tmp = out(reg) _,
            tz = out(reg) _,
            options(pure, nomem, nostack)
        );
    }
    out
}

/// Stein's binary GCD on `u128`, dispatching to `gcd_binary_u64` once both fit in `u64`.
#[inline]
pub const fn gcd_binary_u128(mut a: u128, mut b: u128) -> u128 {
    if a <= (u64::MAX as u128) && b <= (u64::MAX as u128) {
        return gcd_binary_u64(a as u64, b as u64) as u128;
    }
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        if a <= (u64::MAX as u128) && b <= (u64::MAX as u128) {
            return (gcd_binary_u64(a as u64, b as u64) as u128) << shift;
        }
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

impl Ratio {
    pub const ZERO: Self = Self { p: 0, q: 1 };
    pub const ONE: Self = Self { p: 1, q: 1 };
    pub const BONOBO_RESONANCE: Self = BONOBO_RESONANCE;
    pub const ASYMMETRIC_CRITICAL_WEIGHT: Self = ASYMMETRIC_CRITICAL_WEIGHT;

    /// Construct a reduced canonical rational `p / q` (`q == 0` maps to `0/1`).
    #[inline]
    pub const fn new(p: u64, q: u64) -> Self {
        if p == 0 || q == 0 {
            return Self::ZERO;
        }
        if p == q {
            return Self::ONE;
        }
        let g = gcd_binary_u64(p, q);
        if g == 1 {
            return Self { p, q };
        }
        Self { p: p / g, q: q / g }
    }

    /// Construct a reduced canonical rational `p / q` using runtime AArch64 ASM Stein GCD.
    #[inline]
    pub fn new_fast(p: u64, q: u64) -> Self {
        if p == 0 || q == 0 {
            return Self::ZERO;
        }
        if p == q {
            return Self::ONE;
        }
        #[cfg(target_arch = "aarch64")]
        let g = gcd_binary_u64_asm(p, q);
        #[cfg(not(target_arch = "aarch64"))]
        let g = gcd_binary_u64(p, q);
        if g == 1 {
            return Self { p, q };
        }
        Self { p: p / g, q: q / g }
    }

    /// Construct from `u128` intermediates, narrowing after GCD reduction when possible.
    #[inline]
    pub const fn from_u128(p: u128, q: u128) -> Self {
        if p == 0 || q == 0 {
            return Self::ZERO;
        }
        if p <= (u64::MAX as u128) && q <= (u64::MAX as u128) {
            return Self::new(p as u64, q as u64);
        }
        let g = gcd_binary_u128(p, q);
        let (rp, rq) = if g == 1 { (p, q) } else { (p / g, q / g) };
        if rp <= (u64::MAX as u128) && rq <= (u64::MAX as u128) {
            return Self {
                p: rp as u64,
                q: rq as u64,
            };
        }
        let mut sp = rp;
        let mut sq = rq;
        while sp > (u64::MAX as u128) || sq > (u64::MAX as u128) {
            sp >>= 1;
            sq >>= 1;
            if sq == 0 {
                sq = 1;
                break;
            }
        }
        Self::new(sp as u64, sq as u64)
    }

    /// Exact 128-bit cross-multiplication comparison without floating point.
    #[inline]
    pub const fn cross_cmp(self, other: Self) -> Ordering {
        let lhs = (self.p as u128) * (other.q as u128);
        let rhs = (other.p as u128) * (self.q as u128);
        if lhs < rhs {
            Ordering::Less
        } else if lhs > rhs {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Exact rational addition `self + other`.
    #[inline]
    pub const fn add_exact(self, other: Self) -> Self {
        if self.q == 0 || other.q == 0 {
            return Self::ZERO;
        }
        let num = (self.p as u128) * (other.q as u128) + (other.p as u128) * (self.q as u128);
        let den = (self.q as u128) * (other.q as u128);
        Self::from_u128(num, den)
    }

    /// Exact saturating rational subtraction `max(0, self - other)`.
    #[inline]
    pub const fn sub_exact(self, other: Self) -> Self {
        if self.q == 0 || other.q == 0 {
            return Self::ZERO;
        }
        let lhs = (self.p as u128) * (other.q as u128);
        let rhs = (other.p as u128) * (self.q as u128);
        if lhs <= rhs {
            return Self::ZERO;
        }
        let num = lhs - rhs;
        let den = (self.q as u128) * (other.q as u128);
        Self::from_u128(num, den)
    }

    /// Exact rational multiplication `self * other`.
    #[inline]
    pub const fn mul_exact(self, other: Self) -> Self {
        if self.q == 0 || other.q == 0 {
            return Self::ZERO;
        }
        let num = (self.p as u128) * (other.p as u128);
        let den = (self.q as u128) * (other.q as u128);
        Self::from_u128(num, den)
    }

    /// Exact rational division `self / other` (`other.p == 0` maps to `0/1`).
    #[inline]
    pub const fn div_exact(self, other: Self) -> Self {
        if self.q == 0 || other.p == 0 || other.q == 0 {
            return Self::ZERO;
        }
        let num = (self.p as u128) * (other.q as u128);
        let den = (self.q as u128) * (other.p as u128);
        Self::from_u128(num, den)
    }

    /// Project onto Q16.16 unsigned integer (`(p << 16) / q`, rounded down, saturating on `u64`).
    #[inline]
    pub const fn to_q16_u64(self) -> u64 {
        if self.q == 0 {
            return 0;
        }
        let scaled = ((self.p as u128) << Q16_SHIFT) / (self.q as u128);
        if scaled > (u64::MAX as u128) {
            u64::MAX
        } else {
            scaled as u64
        }
    }

    /// Integer part and two decimal digits `(whole, hundredths)` purely via integer math.
    #[inline]
    pub const fn to_decimal_parts_x100(self) -> (u64, u64) {
        if self.q == 0 {
            return (0, 0);
        }
        let scaled = ((self.p as u128) * 100) / (self.q as u128);
        ((scaled / 100) as u64, (scaled % 100) as u64)
    }

    /// Evaluate Ultra-Zeno witness `(P / Q) - n * t > P` in exact `i128` arithmetic.
    #[inline]
    pub const fn ultra_zeno_witness(self, n: u64, t: i8) -> (bool, i128) {
        let p = self.p as i128;
        let q = self.q as i128;
        let lhs_scaled = p - (n as i128) * (t as i128) * q;
        let rhs_scaled = p * q;
        let surplus = lhs_scaled - rhs_scaled;
        (surplus > 0, surplus)
    }
}

impl Ord for Ratio {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cross_cmp(*other)
    }
}

impl PartialOrd for Ratio {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.p, self.q)
    }
}

/// Signed Q15.16 multiplication `(a * b) >> 16`, saturating to `[i32::MIN, i32::MAX]`.
#[inline]
pub const fn q16_mul(a: i32, b: i32) -> i32 {
    let prod = ((a as i64) * (b as i64)) >> Q16_SHIFT;
    if prod > (i32::MAX as i64) {
        i32::MAX
    } else if prod < (i32::MIN as i64) {
        i32::MIN
    } else {
        prod as i32
    }
}

/// Signed Q15.16 division `(a << 16) / b`, returning `0` on `b == 0` and saturating to `i32`.
#[inline]
pub const fn q16_div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let quot = ((a as i64) << Q16_SHIFT) / (b as i64);
    if quot > (i32::MAX as i64) {
        i32::MAX
    } else if quot < (i32::MIN as i64) {
        i32::MIN
    } else {
        quot as i32
    }
}

/// Deterministic SplitMix64 PRNG for zero-float probing and test vectors.
#[inline]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
