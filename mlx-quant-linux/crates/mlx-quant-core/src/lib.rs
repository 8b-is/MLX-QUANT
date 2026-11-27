//! `mlx-quant-core`: `#![no_std]` zero-float integer substrate for MLX-QUANT.
//!
//! Provides exact coprime `Ratio` & Q15.16 fixed-point arithmetic, lock-free epoch
//! bump arenas (`EpochBumpArena` / `EpochBumpArenaSlice`), balanced-ternary / BITCOS /
//! SPQT Base-243 / INT4 / H-BitLinear scalar oracles, zero-`Vec` tensor views, and
//! BeeHive primitives (Tequila STE, Engram, mHC Sinkhorn, Ramanujan, Mushroom-Body WTA).

#![cfg_attr(not(test), no_std)]

pub mod arena;
pub mod beehive;
pub mod ratio;
pub mod tensor;
pub mod ternary;

pub use arena::*;
pub use beehive::*;
pub use ratio::*;
pub use tensor::*;
pub use ternary::*;

#[cfg(test)]
mod tests;
