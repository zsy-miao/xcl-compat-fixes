//! Short names for the `dx.op` operations the direct compute translator uses
//! (the helpers themselves are [`super::ops`]).

/// Float unary operations by their DXIL names.
pub mod op {
    pub use super::super::ops::FUnary;
    pub const SATURATE: FUnary = FUnary::Saturate;
    pub const COS: FUnary = FUnary::Cos;
    pub const SIN: FUnary = FUnary::Sin;
    pub const EXP: FUnary = FUnary::Exp;
    pub const FRC: FUnary = FUnary::Frc;
    pub const LOG: FUnary = FUnary::Log;
    pub const SQRT: FUnary = FUnary::Sqrt;
    pub const RSQRT: FUnary = FUnary::Rsqrt;
    pub const ROUND_NE: FUnary = FUnary::RoundNe;
    pub const ROUND_NI: FUnary = FUnary::RoundNi;
    pub const ROUND_PI: FUnary = FUnary::RoundPi;
    pub const ROUND_Z: FUnary = FUnary::RoundZ;
}
