//! GPU side of XCL: console shader capture and the RDNA2 microcode toolchain
//! (decoder / disassembler – the front-end of the shader recompiler).

pub mod adapter;
pub mod agility;
pub mod bvh;
pub mod desc;
pub mod dxbc;
pub mod dxil;
pub mod effects;
pub(crate) mod fence;
pub mod pack;
pub mod rdna2;
pub mod recomp;
mod rdna2_spec;
mod rdna2_xbox;
pub mod shader;
pub mod texture;
pub mod tiling;
mod tiling_tables;
pub mod uma;
pub mod xbox;

pub use rdna2_spec::ARCH as RDNA2_ARCH;
