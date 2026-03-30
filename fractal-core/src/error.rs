use core::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum FractalError {
    Shape(String),
    InvalidState(String),
    InvalidConfig(String),
}

impl Display for FractalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "shape error: {message}"),
            Self::InvalidState(message) => write!(f, "invalid state: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid config: {message}"),
        }
    }
}

impl std::error::Error for FractalError {}
