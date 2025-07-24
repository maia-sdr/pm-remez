use ndarray::{Array, Dimension};

/// Conversion to a LAPACK-compatible scalar.
///
/// The Parks-McClellan implementation in this crate is written for any scalar
/// type that implements the [`Float`](num_traits::Float) and
/// [`FloatConst`](num_traits::FloatConst) traits. However, for Chebyshev proxy
/// root finding, LAPACK is used through the [`ndarray_linalg`] crate to
/// compute eigenvalues. Therefore, at this point the scalars used by the
/// Parks-McClellan algorithm must be converted to one of the scalar types
/// supported by LAPACK (essentially [`f64`] and [`f32`]). This trait defines
/// this conversion.
pub trait ToLapack: Sized + 'static {
    /// The LAPACK-supported type to which the scalar is converted.
    type Lapack: ndarray_linalg::Lapack;

    /// Returns the conversion of `self` to a `Self::Lapack` scalar.
    fn to_lapack(&self) -> Self::Lapack;

    /// Returns the conversion of a `Self::Lapack` scalar to a `Self` scalar.
    fn from_lapack(lapack: &Self::Lapack) -> Self;

    /// Converts an [`ndarray`] [`Array`] of scalars to an array of
    /// LAPACK-compatible scalars.
    ///
    /// An implementation is provided where [`Array::map`] is used to call the
    /// [`ToLapack::to_lapack`] method for each element of the
    /// array. Implementors of the trait can override this method with a more
    /// efficient implementation.
    fn array_to_lapack<D: Dimension>(array: Array<Self, D>) -> Array<Self::Lapack, D> {
        array.map(|s| s.to_lapack())
    }
}

/// Marker trait used to mark for which types that have the trait [`Lapack`](ndarray_linalg::Lapack),
/// the trait [`ToLapack`] should be implemented as a no-op conversion.
///
/// A marker trait is needed because since `Lapack` is defined by an upstream
/// crate, an attempt to use a blanket implementation
///
/// ```ignore
/// impl<T: ndarray_linalg::Lapack> ToLapack for T { ... }
/// ```
///
/// would be rejected by the compiler because the upstream crate that defines
/// `Lapack` can add implementations of `ToLapack` for specific types, and
/// such implementations would conflict with this blanket implementation.
pub trait IsLapack: ndarray_linalg::Lapack {}
impl IsLapack for f64 {}
impl IsLapack for f32 {}

impl<T: IsLapack> ToLapack for T {
    type Lapack = Self;

    fn to_lapack(&self) -> Self::Lapack {
        *self
    }

    fn from_lapack(lapack: &Self::Lapack) -> Self {
        *lapack
    }

    fn array_to_lapack<D: Dimension>(array: Array<Self, D>) -> Array<Self::Lapack, D> {
        array
    }
}

#[cfg(feature = "num-bigfloat")]
impl ToLapack for num_bigfloat::BigFloat {
    type Lapack = f64;

    fn to_lapack(&self) -> Self::Lapack {
        self.to_f64()
    }

    fn from_lapack(lapack: &Self::Lapack) -> Self {
        Self::from_f64(*lapack)
    }
}
