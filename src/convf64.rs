/// Conversion between a type and [`f64`].
///
/// This trait defines conversions between a type and [`f64`]. The conversions
/// are allowed to be lossy. The trait is mainly intended as a helper when using
/// bigfloats and other non-standard floating point types with some operations
/// that only support `f64`.
pub trait Convf64 {
    /// Converts self to an `f64`, possibly in a lossy way.
    fn to_f64(&self) -> f64;

    /// Converts an `f64` to the type `Self`, possibly in a lossy way.
    fn from_f64(x: f64) -> Self;
}

impl Convf64 for f32 {
    fn to_f64(&self) -> f64 {
        (*self).into()
    }

    fn from_f64(x: f64) -> f32 {
        x as f32
    }
}

impl Convf64 for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn from_f64(x: f64) -> f64 {
        x
    }
}

#[cfg(feature = "num-bigfloat")]
impl Convf64 for num_bigfloat::BigFloat {
    fn to_f64(&self) -> f64 {
        self.to_f64()
    }

    fn from_f64(x: f64) -> Self {
        x.into()
    }
}
