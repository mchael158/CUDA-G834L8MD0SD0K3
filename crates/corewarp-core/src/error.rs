use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Oom,
    Backend(String),
    Busy,
    NotReady,
    Failed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) => write!(f, "invalid: {msg}"),
            Self::Oom => write!(f, "out of memory"),
            Self::Backend(msg) => write!(f, "backend: {msg}"),
            Self::Busy => write!(f, "backpressure: resources busy"),
            Self::NotReady => write!(f, "not ready"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
