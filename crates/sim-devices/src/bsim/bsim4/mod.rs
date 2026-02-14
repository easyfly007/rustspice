//! BSIM4-specific physics modules
//!
//! This module contains physics models unique to BSIM4 (Level 54):
//! - Impact ionization (substrate current)
//! - Layout-dependent stress effects
//! - Gate tunneling current

pub mod substrate;
pub mod stress;
pub mod tunneling;

pub use substrate::calculate_isub;
pub use stress::calculate_stress_effects;
pub use tunneling::calculate_gate_tunneling;
