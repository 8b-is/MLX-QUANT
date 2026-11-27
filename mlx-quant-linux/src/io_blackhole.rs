//! Bounded zero-panic reader portal loading tensor bytes directly into [`BumpAllocator`].

use crate::allocator::BumpAllocator;
use crate::tensor::{DType, Shape, Tensor};
use std::fs::File;
use std::io::{self, Read};

/// Bounded tensor loader into an epoch-scoped [`BumpAllocator`] arena.
pub struct IoBlackhole<'a> {
    allocator: &'a BumpAllocator,
}

impl<'a> IoBlackhole<'a> {
    /// Construct an [`IoBlackhole`] bound to `allocator`.
    #[inline]
    pub const fn new(allocator: &'a BumpAllocator) -> Self {
        Self { allocator }
    }

    /// Read at most `tensor.byte_len()` bytes from `reader` into a freshly allocated [`Tensor`].
    ///
    /// Short reads and early EOF are handled safely: unread trailing bytes are deterministically
    /// zero-padded, and `(tensor, bytes_read, fnv1a_digest)` is returned.
    pub fn load_bytes_from_reader(
        &self,
        reader: &mut impl Read,
        shape: Shape,
        dtype: DType,
    ) -> io::Result<(Tensor<'a>, usize, u64)> {
        let mut tensor = Tensor::new(shape, dtype, self.allocator).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "BumpAllocator exhausted or invalid tensor shape",
            )
        })?;

        let buf = tensor.as_bytes_mut();
        let cap = buf.len();
        let mut total_read = 0usize;

        while total_read < cap {
            match reader.read(&mut buf[total_read..]) {
                Ok(0) => break,
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }

        if total_read < cap {
            buf[total_read..].fill(0);
        }

        let digest = tensor.fnv1a_digest();
        Ok((tensor, total_read, digest))
    }

    /// Bounded read from an open [`std::fs::File`] directly into an arena-allocated [`Tensor`].
    #[inline]
    pub fn suck_weights_from_disk(
        &self,
        file: &mut File,
        shape: Shape,
        dtype: DType,
    ) -> io::Result<(Tensor<'a>, usize, u64)> {
        self.load_bytes_from_reader(file, shape, dtype)
    }
}
