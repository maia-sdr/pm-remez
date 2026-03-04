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

#[cfg(any(feature = "lapack-backend", feature = "nalgebra-backend"))]
macro_rules! default_eigenvalue_doc {
    () => {
        r#" Default eigenvalue backend.

 This defines the default eigenvalue backend, which depends on what feature
 flags are enabled. The selected default backend is the first available from this priority list:

 - `lapack-backend`
 - `nalgebra-backend`
"#
    };
}

#[doc = default_eigenvalue_doc!()]
#[cfg(feature = "lapack-backend")]
pub type DefaultEigenvalueBackend = LapackBackend;

#[doc = default_eigenvalue_doc!()]
#[cfg(all(not(any(feature = "lapack-backend")), feature = "nalgebra-backend"))]
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
    /// `num_bigfloat::BigFloat` the type is converted first to `f64`.
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
