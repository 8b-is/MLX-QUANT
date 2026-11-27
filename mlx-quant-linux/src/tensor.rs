//! Zero-`Vec`, zero-float tensor handle backed by [`BumpAllocator`] or borrowed raw slices.

use crate::allocator::BumpAllocator;
use crate::fnv1a_64;
use core::mem::align_of;
pub use mlx_quant_core::tensor::{DType, Shape, TensorError, TensorMut, TensorView};

/// Compute checked byte allocation size for `numel` elements of `dtype` without saturating overflow.
#[inline]
pub(crate) fn checked_bytes_for_numel(dtype: DType, numel: usize) -> Option<usize> {
    match dtype {
        DType::Ternary2Bit => Some(numel.div_ceil(4)),
        DType::BitcosSparse => 2usize.checked_mul(numel.div_ceil(8)),
        DType::SpqtBase243 => Some(numel.div_ceil(5)),
        DType::Int4Packed => Some(numel.div_ceil(2)),
        DType::Int8 => Some(numel),
        DType::Int32 | DType::Q15_16 => numel.checked_mul(4),
        DType::Float16Raw => numel.checked_mul(2),
    }
}

/// Compute checked byte allocation size for a shaped tensor, using row-strided byte alignment
/// on the innermost dimension for sub-byte packed formats (`Ternary2Bit`, `Int4Packed`, etc.).
#[inline]
pub(crate) fn checked_bytes_for_shape(dtype: DType, shape: Shape) -> Option<usize> {
    let numel = shape.checked_numel()?;
    if shape.ndim <= 1 {
        return checked_bytes_for_numel(dtype, numel);
    }
    let last_dim = shape.dims[shape.ndim as usize - 1];
    if numel == 0 || last_dim == 0 {
        return Some(0);
    }
    let outer = numel / last_dim;
    let row_bytes = checked_bytes_for_numel(dtype, last_dim)?;
    outer.checked_mul(row_bytes)
}

/// Epoch-scoped quantized/integer tensor backed by `&'a mut [u8]` from [`BumpAllocator`].
#[derive(Debug)]
pub struct Tensor<'a> {
    shape: Shape,
    dtype: DType,
    data: &'a mut [u8],
    allocator: Option<&'a BumpAllocator>,
    slot: Option<u8>,
}

impl<'a> Tensor<'a> {
    /// Allocate a 64-byte-aligned tensor from `allocator`, claiming an optional tracking slot.
    pub fn new(shape: Shape, dtype: DType, allocator: &'a BumpAllocator) -> Option<Self> {
        let alloc_size = checked_bytes_for_shape(dtype, shape)?;
        let data = allocator.alloc_tensor_bytes(alloc_size)?;
        let slot = allocator.claim_slot();
        Some(Self {
            shape,
            dtype,
            data,
            allocator: Some(allocator),
            slot,
        })
    }

    /// Wrap a caller-provided mutable buffer after verifying shape and minimum byte length.
    pub fn new_raw(data: &'a mut [u8], shape: Shape, dtype: DType) -> Option<Self> {
        let numel = shape.checked_numel()?;
        let min_bytes = if dtype == DType::BitcosSparse {
            dtype.min_bytes_for_numel(numel)
        } else {
            checked_bytes_for_shape(dtype, shape)?
        };
        if data.len() < min_bytes {
            return None;
        }
        Some(Self {
            shape,
            dtype,
            data,
            allocator: None,
            slot: None,
        })
    }

    /// Tensor shape descriptor (`1..=4` dimensions).
    #[inline]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    /// Quantized or integer element dtype.
    #[inline]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total logical element count (`shape.numel()`).
    #[inline]
    pub const fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Physical backing storage size in bytes.
    #[inline]
    pub const fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Optional allocator tracking slot ID (`0..64`) held until `Drop`.
    #[inline]
    pub const fn slot(&self) -> Option<u8> {
        self.slot
    }

    /// Immutable byte view over the tensor storage.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Mutable byte view over the tensor storage.
    #[inline]
    pub const fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Checked immutable [`TensorView`] from `mlx-quant-core`.
    #[inline]
    pub fn as_view(&self) -> Result<TensorView<'_>, TensorError> {
        TensorView::new_checked(self.data, self.shape, self.dtype)
    }

    /// Checked mutable [`TensorMut`] from `mlx-quant-core`.
    #[inline]
    pub fn as_view_mut(&mut self) -> Result<TensorMut<'_>, TensorError> {
        TensorMut::new_checked(self.data, self.shape, self.dtype)
    }

    /// View `DType::Int8` backing storage as `&[i8]`.
    #[inline]
    pub fn as_i8_slice(&self) -> Option<&[i8]> {
        if self.dtype != DType::Int8 {
            return None;
        }
        let numel = self.shape.checked_numel()?;
        if self.data.len() < numel {
            return None;
        }
        if numel == 0 {
            return Some(&[]);
        }
        // SAFETY: `self.dtype == DType::Int8`, `i8` has alignment 1 and size 1, and
        // `self.data.len() >= numel` guarantees every byte in `0..numel` is initialized.
        Some(unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast::<i8>(), numel) })
    }

    /// View `DType::Int8` backing storage as `&mut [i8]`.
    #[inline]
    pub fn as_i8_slice_mut(&mut self) -> Option<&mut [i8]> {
        if self.dtype != DType::Int8 {
            return None;
        }
        let numel = self.shape.checked_numel()?;
        if self.data.len() < numel {
            return None;
        }
        if numel == 0 {
            return Some(&mut []);
        }
        // SAFETY: `self.dtype == DType::Int8`, `i8` has alignment 1 and size 1, and `self.data`
        // is exclusively borrowed for `numel` bytes.
        Some(unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<i8>(), numel) })
    }

    /// View `DType::Int32` or `DType::Q15_16` backing storage as `&[i32]`.
    #[inline]
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.dtype != DType::Int32 && self.dtype != DType::Q15_16 {
            return None;
        }
        let numel = self.shape.checked_numel()?;
        let needed = numel.checked_mul(4)?;
        if self.data.len() < needed {
            return None;
        }
        if numel == 0 {
            return Some(&[]);
        }
        if !(self.data.as_ptr() as usize).is_multiple_of(align_of::<i32>()) {
            return None;
        }
        // SAFETY: `self.dtype` is `Int32`/`Q15_16`, pointer is 4-byte aligned, and `self.data`
        // holds at least `numel * 4` initialized bytes.
        Some(unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast::<i32>(), numel) })
    }

    /// View `DType::Int32` or `DType::Q15_16` backing storage as `&mut [i32]`.
    #[inline]
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        if self.dtype != DType::Int32 && self.dtype != DType::Q15_16 {
            return None;
        }
        let numel = self.shape.checked_numel()?;
        let needed = numel.checked_mul(4)?;
        if self.data.len() < needed {
            return None;
        }
        if numel == 0 {
            return Some(&mut []);
        }
        if !(self.data.as_ptr() as usize).is_multiple_of(align_of::<i32>()) {
            return None;
        }
        // SAFETY: `self.dtype` is `Int32`/`Q15_16`, pointer is 4-byte aligned, and `self.data`
        // is uniquely borrowed across at least `numel * 4` bytes.
        Some(unsafe {
            core::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<i32>(), numel)
        })
    }

    /// Zero-fill all backing bytes safely.
    #[inline]
    pub fn zero(&mut self) {
        self.data.fill(0);
    }

    /// Deterministic 64-bit FNV-1a integrity digest of the tensor's backing bytes.
    #[inline]
    pub fn fnv1a_digest(&self) -> u64 {
        fnv1a_64(self.data)
    }
}

impl<'a> Drop for Tensor<'a> {
    fn drop(&mut self) {
        if let (Some(allocator), Some(slot)) = (self.allocator, self.slot) {
            let _ = allocator.release_slot(slot);
        }
    }
}
