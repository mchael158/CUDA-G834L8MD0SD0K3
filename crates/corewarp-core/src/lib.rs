//! Tipos e erros do CoreWarp (sem dependência CUDA).

mod error;
mod types;

pub use error::{Error, Result};
pub use types::*;
