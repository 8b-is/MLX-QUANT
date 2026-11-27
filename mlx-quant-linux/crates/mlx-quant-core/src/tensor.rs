//! Zero-`Vec`, `#![no_std]` tensor shapes, quantized dtypes, and checked byte views.

/// Supported tensor element / packed quantization formats in `mlx-quant`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    /// 2-bit packed ternary (`4 trits/byte`).
    Ternary2Bit,
    /// BITCOS sparse presence-bitmap + compacted sign stream (`1..=2 bpw`, worst-case `2 * ceil(N/8)`).
    BitcosSparse,
    /// SPQT Base-243 (`5 trits/byte`, `1.60 bpw` payload).
    SpqtBase243,
    /// Symmetric two's-complement INT4 (`2 nibbles/byte`, low nibble first).
    Int4Packed,
    /// Signed 8-bit integer (`1 byte/elem`).
    Int8,
    /// Signed 32-bit integer (`4 bytes/elem`).
    Int32,
    /// Signed Q15.16 fixed-point integer (`4 bytes/elem`).
    Q15_16,
    /// Raw IEEE-754 binary16 bit pattern (`u16`, `2 bytes/elem`, zero host float arithmetic).
    Float16Raw,
}

impl DType {
    /// Nominal buffer capacity in bytes required to store `numel` elements of this dtype.
    ///
    /// For [`DType::BitcosSparse`], returns the worst-case upper bound `2 * numel.div_ceil(8)`
    /// (full presence bitmap + dense sign stream).
    #[inline]
    pub const fn bytes_for_numel(self, numel: usize) -> usize {
        match self {
            Self::Ternary2Bit => numel.div_ceil(4),
            Self::BitcosSparse => 2usize.saturating_mul(numel.div_ceil(8)),
            Self::SpqtBase243 => numel.div_ceil(5),
            Self::Int4Packed => numel.div_ceil(2),
            Self::Int8 => numel,
            Self::Int32 | Self::Q15_16 => numel.saturating_mul(4),
            Self::Float16Raw => numel.saturating_mul(2),
        }
    }

    /// Minimum valid buffer size in bytes for `numel` elements (presence-only floor for BITCOS).
    #[inline]
    pub const fn min_bytes_for_numel(self, numel: usize) -> usize {
        match self {
            Self::BitcosSparse => numel.div_ceil(8),
            _ => self.bytes_for_numel(numel),
        }
    }

    /// Canonical short name for telemetry and diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ternary2Bit => "ternary_2bit",
            Self::BitcosSparse => "bitcos_sparse",
            Self::SpqtBase243 => "spqt_base243",
            Self::Int4Packed => "int4_packed",
            Self::Int8 => "i8",
            Self::Int32 => "i32",
            Self::Q15_16 => "q15_16",
            Self::Float16Raw => "f16_raw",
        }
    }
}

/// Fixed-rank (`1..=4`) stack-allocated tensor shape (`#![no_std]`, zero `Vec`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
    pub ndim: u8,
}

impl Shape {
    /// Construct a 1D tensor shape `[d0]`.
    #[inline]
    pub const fn new_1d(d0: usize) -> Self {
        Self {
            dims: [d0, 1, 1, 1],
            ndim: 1,
        }
    }

    /// Construct a 2D tensor shape `[d0, d1]`.
    #[inline]
    pub const fn new_2d(d0: usize, d1: usize) -> Self {
        Self {
            dims: [d0, d1, 1, 1],
            ndim: 2,
        }
    }

    /// Construct a shape from a slice of `1..=4` dimensions.
    #[inline]
    pub const fn from_slice(dims_in: &[usize]) -> Option<Self> {
        if dims_in.is_empty() || dims_in.len() > 4 {
            return None;
        }
        let mut dims = [1usize; 4];
        let mut i = 0usize;
        while i < dims_in.len() {
            dims[i] = dims_in[i];
            i += 1;
        }
        Some(Self {
            dims,
            ndim: dims_in.len() as u8,
        })
    }

    /// Checked total element count, returning `None` on invalid `ndim` or `usize` overflow.
    #[inline]
    pub const fn checked_numel(&self) -> Option<usize> {
        if self.ndim == 0 || self.ndim > 4 {
            return None;
        }
        let mut prod = 1usize;
        let mut i = 0usize;
        let n = self.ndim as usize;
        while i < n {
            let Some(next) = prod.checked_mul(self.dims[i]) else {
                return None;
            };
            prod = next;
            i += 1;
        }
        Some(prod)
    }

    /// Total element count (saturating on overflow; `0` if `ndim == 0`).
    #[inline]
    pub const fn numel(&self) -> usize {
        if self.ndim == 0 || self.ndim > 4 {
            return 0;
        }
        let mut prod = 1usize;
        let mut i = 0usize;
        let n = self.ndim as usize;
        while i < n {
            prod = prod.saturating_mul(self.dims[i]);
            i += 1;
        }
        prod
    }

    /// Borrow active dimensions `&dims[..ndim]`.
    #[inline]
    pub const fn as_slice(&self) -> &[usize] {
        let n = if (self.ndim as usize) > 4 {
            4
        } else {
            self.ndim as usize
        };
        let (head, _) = self.dims.split_at(n);
        head
    }
}

/// Validation failure returned by [`TensorView::new_checked`] and [`TensorMut::new_checked`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    InvalidShape,
    BufferTooSmall { required: usize, actual: usize },
}

/// Immutable checked view over a quantized or integer tensor buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TensorView<'a> {
    data: &'a [u8],
    shape: Shape,
    dtype: DType,
}

impl<'a> TensorView<'a> {
    /// Construct a validated [`TensorView`], ensuring `shape` is valid and `data` has
    /// sufficient bytes for `shape.numel()` in `dtype`.
    #[inline]
    pub fn new_checked(data: &'a [u8], shape: Shape, dtype: DType) -> Result<Self, TensorError> {
        let numel = shape.checked_numel().ok_or(TensorError::InvalidShape)?;
        let required = dtype.min_bytes_for_numel(numel);
        if data.len() < required {
            return Err(TensorError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape, dtype })
    }

    #[inline]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    #[inline]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    #[inline]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub const fn numel(&self) -> usize {
        self.shape.numel()
    }
}

/// Mutable checked view over a quantized or integer tensor buffer.
#[derive(Debug, PartialEq, Eq)]
pub struct TensorMut<'a> {
    data: &'a mut [u8],
    shape: Shape,
    dtype: DType,
}

impl<'a> TensorMut<'a> {
    /// Construct a validated [`TensorMut`], ensuring `shape` is valid and `data` has
    /// sufficient bytes for `shape.numel()` in `dtype`.
    #[inline]
    pub fn new_checked(data: &'a mut [u8], shape: Shape, dtype: DType) -> Result<Self, TensorError> {
        let numel = shape.checked_numel().ok_or(TensorError::InvalidShape)?;
        let required = dtype.min_bytes_for_numel(numel);
        if data.len() < required {
            return Err(TensorError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape, dtype })
    }

    #[inline]
    pub const fn data(&self) -> &[u8] {
        self.data
    }

    #[inline]
    pub const fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    #[inline]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    #[inline]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub const fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Reborrow as an immutable [`TensorView`].
    #[inline]
    pub const fn as_view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data,
            shape: self.shape,
            dtype: self.dtype,
        }
    }
}
