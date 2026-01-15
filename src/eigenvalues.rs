use crate::error::Error;
use ndarray::Array2;
use num_complex::Complex;

/// Eigenvalue backend.
///
/// This trait models a backend that performs the computation of the eigenvalues
/// of square matrices with real scalars of type `T`.
///
/// `pm-remez` includes support for several Rust eigenvalue librariers through
/// types that implement this trait. These libraries are optional and selected
/// with features flags. At the moment the following backends are supported:
///
/// - `lapack-backend` feature flag. This defines the `LapackBackend` backend,
///   which uses `ndarray_linalg` to compute eigenvalues with LAPACK.
///
/// - `faer-backend` feature flag. This defines the `FaerBackend` backend,
///   which uses `faer` to compute eigenvalues.
///
/// - `nalgebra-backend` feature flag. This defines the `NalgebraBackend`,
///   which uses `nalgebra` to compute eigenvalues.
pub trait EigenvalueBackend<T> {
    /// Computes the eigenvalues of a real square matrix with scalar type `T`.
    ///
    /// An error is returned if the eigenvalues cannot be computed.
    ///
    /// # Panics
    ///
    /// This function is allowed to panic if `matrix` is not a square matrix.
    fn eigenvalues(&self, matrix: Array2<T>) -> Result<Vec<Complex<T>>>;
}

type Result<T> = std::result::Result<T, EigenvaluesError>;

/// Eigenvalue calculation error.
///
/// This struct represents an error obtained by an eigenvalue backend during an
/// eigenvalue compuation. The error contains a descriptive string of the
/// problem.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct EigenvaluesError(pub String);

impl From<EigenvaluesError> for Error {
    fn from(value: EigenvaluesError) -> Error {
        Error::EigenvaluesError(value.0)
    }
}

#[cfg(any(
    feature = "lapack-backend",
    feature = "faer-backend",
    feature = "nalgebra-backend"
))]
macro_rules! default_eigenvalue_doc {
    () => {
        r#" Default eigenvalue backend.

 This defines the default eigenvalue backend, which depends on what feature
 flags are enabled. The selected default backend is the first available from this priority list:

 - `lapack-backend`
 - `faer-backend`
 - `nalgebra-backend`
"#
    };
}

#[doc = default_eigenvalue_doc!()]
#[cfg(feature = "lapack-backend")]
pub type DefaultEigenvalueBackend = LapackBackend;

#[doc = default_eigenvalue_doc!()]
#[cfg(all(not(feature = "lapack-backend"), feature = "faer-backend"))]
pub type DefaultEigenvalueBackend = FaerBackend;

#[doc = default_eigenvalue_doc!()]
#[cfg(all(
    not(any(feature = "lapack-backend", feature = "faer-backend")),
    feature = "nalgebra-backend"
))]
pub type DefaultEigenvalueBackend = NalgebraBackend;

#[cfg(feature = "lapack-backend")]
pub use lapack::LapackBackend;

#[cfg(feature = "lapack-backend")]
mod lapack {
    use super::*;
    use crate::lapack::ToLapack;
    use ndarray_linalg::{EigVals, Scalar, error::LinalgError};

    /// LAPACK eigenvalue backend.
    ///
    /// This is an eigenvalue backend that uses [`ndarray_linalg`] to compute
    /// eigenvalues with LAPACK. For types natively supported by LAPACK, which
    /// are `f64` and `f32`, the calculations are done directly using that
    /// type. For other types, the [`ToLapack`] trait is used to convert the
    /// type `T` into a type (generally `f64`) that can be handled by LAPACK.
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
    pub struct LapackBackend {}

    impl<T: ToLapack> EigenvalueBackend<T> for LapackBackend {
        fn eigenvalues(&self, matrix: Array2<T>) -> Result<Vec<Complex<T>>> {
            let matrix = T::array_to_lapack(matrix);
            let eig = matrix.eigvals()?;
            Ok(eig
                .into_iter()
                .map(|z| {
                    Complex::new(
                        T::from_lapack(&<T::Lapack>::from_real(z.re())),
                        T::from_lapack(&<T::Lapack>::from_real(z.im())),
                    )
                })
                .collect())
        }
    }

    impl From<LinalgError> for EigenvaluesError {
        fn from(value: LinalgError) -> EigenvaluesError {
            EigenvaluesError(value.to_string())
        }
    }
}

#[cfg(feature = "faer-backend")]
pub use faer::FaerBackend;

#[cfg(feature = "faer-backend")]
mod faer {
    use super::*;
    use ::faer::{linalg::evd::EvdError, traits::RealField};
    use faer_ext::IntoFaer;

    // This is needed because num_bigfloat can be a broken link if the crate is
    // not being built due to selected feature flags.
    #[allow(rustdoc::broken_intra_doc_links)]
    /// faer eigenvalue backend.
    ///
    /// This is an eigenvalue backend that uses [`faer`](::faer) to compute
    /// eigenvalues. For types natively supported by `faer`, the calculations
    /// are done using that type. With [`num_bigfloat::BigFloat`] the type is
    /// converted first to `f64`.
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
    pub struct FaerBackend {}

    /// Marker trait used to mark for which types `T` that have the trait
    /// [`RealField`](RealField), the trait `EigenvalueBacked<T>` should be
    /// implemented for `FaerBackend` by doing no scalar type conversion and
    /// using the type `T` natively in [`faer`].
    ///
    /// A marker trait is needed because since `RealField` is defined by an
    /// upstream crate, because if a blanket implementation
    ///
    /// ```ignore
    /// impl<T: RealField> EigenvalueBackend<T> for FaerBackend { ... }
    /// ```
    ///
    /// was used, then it would not be possible to do specialized
    /// implementations for types `T` that do not implement `RealField` (because
    /// at any point the upstream crate could add an implementation of
    /// `RealField` for these types).
    pub trait IsRealField: RealField {}
    impl IsRealField for f64 {}
    impl IsRealField for f32 {}
    impl IsRealField for ::faer::traits::Symbolic {}
    impl IsRealField for ::faer::fx128 {}

    impl<T: IsRealField> EigenvalueBackend<T> for FaerBackend {
        fn eigenvalues(&self, matrix: Array2<T>) -> Result<Vec<Complex<T>>> {
            let matrix = matrix.view().into_faer();
            let eig = matrix.eigenvalues()?;
            Ok(eig)
        }
    }

    // Stop-gap to use num-bigfloat with faer, by converting back and forth to
    // f64.
    //
    // TODO: add native support for num-bigfloat in faer or use faer's higher
    // precision fx128.
    #[cfg(feature = "num-bigfloat")]
    impl EigenvalueBackend<num_bigfloat::BigFloat> for FaerBackend {
        fn eigenvalues(
            &self,
            matrix: Array2<num_bigfloat::BigFloat>,
        ) -> Result<Vec<Complex<num_bigfloat::BigFloat>>> {
            // convert to f64 faer matrix
            let matrix = matrix.map(|x| x.to_f64());
            let matrix = matrix.view().into_faer();
            let eig = matrix.eigenvalues()?;
            Ok(eig
                .iter()
                .map(|x| {
                    Complex::new(
                        num_bigfloat::BigFloat::from(x.re),
                        num_bigfloat::BigFloat::from(x.im),
                    )
                })
                .collect())
        }
    }

    impl From<EvdError> for EigenvaluesError {
        fn from(value: EvdError) -> EigenvaluesError {
            match value {
                EvdError::NoConvergence => EigenvaluesError("no convergence".to_string()),
            }
        }
    }
}

#[cfg(feature = "nalgebra-backend")]
pub use nalgebra::NalgebraBackend;

#[cfg(feature = "nalgebra-backend")]
mod nalgebra {
    use super::*;
    use ::nalgebra::{DMatrix, RealField};

    /// nalgebra eigenvalue backend.
    ///
    /// This is an eigenvalue backend that uses [`nalgebra`](::nalgebra) to compute
    /// eigenvalues. For types natively supported by `nalgebra`, which are only
    /// `f32` and `f64`, the calculations are done using that type. With
    /// [`num_bigfloat::BigFloat`] the type is converted first to `f64`.
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
    pub struct NalgebraBackend {}

    /// Marker trait used to mark for which types `T` that have the trait
    /// [`RealField`](RealField), the trait `EigenvalueBacked<T>` should be
    /// implemented for `NalgebraBackend` by doing no scalar type conversion and
    /// using the type `T` natively in [`nalgebra`].
    ///
    /// A marker trait is needed because since `RealField` is defined by an
    /// upstream crate, because if a blanket implementation
    ///
    /// ```ignore
    /// impl<T: RealField> EigenvalueBackend<T> for NalgebraBackend { ... }
    /// ```
    ///
    /// was used, then it would not be possible to do specialized
    /// implementations for types `T` that do not implement `RealField` (because
    /// at any point the upstream crate could add an implementation of
    /// `RealField` for these types).
    pub trait IsRealField: RealField {}
    impl IsRealField for f64 {}
    impl IsRealField for f32 {}

    impl<T: IsRealField> EigenvalueBackend<T> for NalgebraBackend {
        fn eigenvalues(&self, matrix: Array2<T>) -> Result<Vec<Complex<T>>> {
            let matrix = DMatrix::from_row_iterator(matrix.nrows(), matrix.ncols(), matrix);
            let eig = matrix.complex_eigenvalues();
            Ok(eig.into_iter().cloned().collect())
        }
    }

    #[cfg(feature = "num-bigfloat")]
    impl EigenvalueBackend<num_bigfloat::BigFloat> for NalgebraBackend {
        fn eigenvalues(
            &self,
            matrix: Array2<num_bigfloat::BigFloat>,
        ) -> Result<Vec<Complex<num_bigfloat::BigFloat>>> {
            // convert to f64 nalgebra matrix
            let matrix = DMatrix::from_row_iterator(
                matrix.nrows(),
                matrix.ncols(),
                matrix.into_iter().map(|x| x.to_f64()),
            );
            let eig = matrix.complex_eigenvalues();
            Ok(eig
                .into_iter()
                .map(|z| {
                    Complex::new(
                        num_bigfloat::BigFloat::from_f64(z.re),
                        num_bigfloat::BigFloat::from_f64(z.im),
                    )
                })
                .collect())
        }
    }
}
