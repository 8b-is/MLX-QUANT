//! Peer-to-peer PCIe BAR DMA tunnel descriptor (`DesignedDescriptor` outside bare-metal Linux).

use crate::ExecutionMode;

/// Pure-integer descriptor of a peer-to-peer PCIe BAR DMA tunnel.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct P2pTunnelDescriptor {
    pub src_bar: usize,
    pub dst_bar: usize,
    pub mtu_bytes: usize,
    pub bar_distance: usize,
    pub mode: ExecutionMode,
}

/// P2P PCIe BAR DMA descriptor builder.
pub struct Wormhole;

impl Wormhole {
    /// Construct a peer-to-peer DMA tunnel descriptor between two PCIe BAR apertures.
    #[inline]
    pub const fn open_p2p_tunnel(
        src_bar: usize,
        dst_bar: usize,
        mtu_bytes: usize,
    ) -> P2pTunnelDescriptor {
        P2pTunnelDescriptor {
            src_bar,
            dst_bar,
            mtu_bytes,
            bar_distance: src_bar.abs_diff(dst_bar),
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
