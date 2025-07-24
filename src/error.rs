//! Error types used by `pm_remez`.

use thiserror::Error;

/// `pm_remez` `Result` type.
pub type Result<T> = core::result::Result<T, Error>;

/// `pm_remez` error.
///
/// This enum represents all the errors that can be produced by `pm_remez`.
#[derive(Error, Debug)]
pub enum Error {
    /// The list of bands is empty.
    #[error("the list of bands is empty")]
    BandsEmpty,
    /// The begin of the band is greater than the end of the band.
    #[error("band begin is greater than band end")]
    BandLimitsWrongOrder,
    /// The band limits are out of bounds.
    #[error("band limits out of bounds")]
    BandLimitsOutOfBounds,
    /// The bands overlap.
    #[error("bands overlap")]
    BandsOverlap,
    /// The derivative of the Chebyshev proxy polynomial is zero.
    ///
    /// This error can happen due to numerical errors, and it prevents the Remez
    /// exchange algorithm from continuing.
    #[error("derivative of Chebyshev proxy is zero")]
    ProxyDerivativeZero,
    /// An error happened during the computation of eigenvalues.
    ///
    /// Eigenvalues are computed to find the roots of the derivative of the
    /// Chebyshev proxy. This error is typically produced by LAPACK, and it can
    /// happen due to numerical errors. It prevents the Remez exchange algorithm
    /// from continuing.
    #[error("unable to compute eigenvalues: {0}")]
    EigenvaluesError(String),
    /// The desired response that was specified is invalid for this type of FIR
    /// filter.
    #[error("invalid desired response: {0}")]
    InvalidResponse(InvalidResponse),
    /// Not enough alternating extrema were found for Remez exchange.
    ///
    /// This error is typically caused by numerical errors.
    #[error("not enough alternating error extrema found")]
    NotEnoughExtrema,
}

/// Invalid desired response error.
///
/// This enum classifies the types of invalid response error that can happen.
#[derive(Error, Debug)]
pub enum InvalidResponse {
    /// An even symmetric even length filter must have zero response at the Nyquist frequency
    #[error(
        "an even symmetric even length filter must have zero response at the Nyquist frequency"
    )]
    EvenSymmEvenLengthNyquist,
    /// An odd symmetric even length filter must have zero response at DC
    #[error("an odd symmetric even length filter must have zero response at DC")]
    OddSymmEvenLengthDC,
    /// An odd symmetric odd length filter must have zero response at DC
    #[error("an odd symmetric odd length filter must have zero response at DC")]
    OddSymmOddLengthDC,
    /// An odd symmetric odd length filter must have zero response at the Nyquist frequency
    #[error("an odd symmetric odd length filter must have zero response at the Nyquist frequency")]
    OddSymmOddLengthNyquist,
}

impl From<InvalidResponse> for Error {
    fn from(value: InvalidResponse) -> Error {
        Error::InvalidResponse(value)
    }
}
