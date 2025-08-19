use crate::error::{Error, Result};
use num_traits::{Float, FloatConst};

/// Band.
///
/// A band defines a closed subinterval of [0.0, 0.5] in which the
/// Parks-McClellan algorithm attempts to make the weighted error function as
/// small as possible.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Band<T> {
    begin: T,
    end: T,
}

impl<T: Float> Band<T> {
    /// Creates a new band.
    ///
    /// The band is the closed interval `[begin, end]`.
    pub fn new(begin: T, end: T) -> Result<Band<T>> {
        if !begin.is_finite() || !end.is_finite() {
            return Err(Error::BandLimitsOutOfBounds);
        }
        if begin > end {
            return Err(Error::BandLimitsWrongOrder);
        }
        if begin < T::zero() || end > T::from(0.5).unwrap() {
            return Err(Error::BandLimitsOutOfBounds);
        }
        Ok(Band { begin, end })
    }
}

impl<T: Copy> Band<T> {
    /// Returns the beginning of the band.
    pub fn begin(&self) -> T {
        self.begin
    }

    /// Returns the end of the band.
    pub fn end(&self) -> T {
        self.end
    }
}

impl<T: Float> Band<T> {
    /// Returns the length of the band.
    ///
    /// The length is defined as `end - begin`.
    pub fn len(&self) -> T {
        self.end - self.begin
    }

    /// Returns true if the band contains a certain element.
    pub fn contains(&self, element: T) -> bool {
        (self.begin()..=self.end()).contains(&element)
    }

    /// Returns true if the two bands overlap.
    pub fn overlaps(&self, other: &Band<T>) -> bool {
        self.end() > other.begin() && other.end() > self.begin()
    }

    /// Returns the distance between an element and the band.
    ///
    /// The distance is defined as zero if the band contains the element, and as
    /// the distance between the element and the closest endpoint of the band
    /// otherwise.
    pub fn distance(&self, element: T) -> T {
        if self.contains(element) {
            T::zero()
        } else {
            (element - self.begin())
                .abs()
                .min((element - self.end()).abs())
        }
    }
}

impl<T: Float + FloatConst> Band<T> {
    pub(super) fn convert_to_radians(&mut self) {
        let pi = T::PI();
        let two_pi = T::TAU();
        // The min() function is to avoid rounding from sending the point beyond
        // pi.
        self.begin = (self.begin * two_pi).min(pi);
        self.end = (self.end * two_pi).min(pi);
    }
}

/// A FIR filter design produced by the [`pm_remez`](crate::pm_remez) function.
///
/// The type parameter `T` corresponds to the scalar type used in the
/// calculations. Typically it implements the [`Float`] trait.
#[derive(Debug, Clone)]
pub struct PMDesign<T> {
    /// Impulse response of the FIR filter.
    ///
    /// This is the list of taps of the filter.
    pub impulse_response: Vec<T>,
    /// Maximum weighted error achieved by the filter.
    pub weighted_error: T,
    /// Extremal frequencies of the filter produced in the final iteration of
    /// the Remez exchange.
    ///
    /// The frequencies are in the interval [0.0, 0.5].
    pub extremal_freqs: Vec<T>,
    /// Number of iterations performed by the Parks-McClellan algorithm.
    pub num_iterations: usize,
    /// Flatness of the solution.
    ///
    /// The flatness is used to assess if the algorithm has converged. It is
    /// defined as the difference between the maximum absolute value of the
    /// weighted error over all the extremal frequencies minus the minimum
    /// absolute value of the weighted error over all the extremal frequencies,
    /// divided by the maximum absolute value of the weighted error over all the
    /// extremal frequencies.
    pub flatness: T,
}

/// Symmetry of the FIR filter taps.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Symmetry {
    /// Even symmetry.
    ///
    /// This is generally used for lowpass, bandpass and highpass filters.
    Even,
    /// Odd symmetry.
    ///
    /// This is generally used for Hilbert and differentiator filters.
    Odd,
}

/// Parks-McClellan design parameters struct.
///
/// This struct is one of the two ways (together with the
/// [`pm_parameters`](super::pm_parameters) function used with a list of
/// [`BandSetting`](super::BandSetting) objects) in which the parameters to
/// execute the Parks-McClellan algorithm can be designed. This struct specifies
/// custom closures for the desired response and weight function, represented by
/// the type parameters `D` and `W`. The type parameter `T` is the scalar type
/// used for computation. Typically it should implement the [`Float`] and
/// [`FloatConst`] traits.
pub struct PMParameters<T, D, W> {
    num_taps: usize,
    bands: Vec<Band<T>>,
    symmetry: Symmetry,
    desired_response: D,
    weights: W,
    cheby_proxy_m: usize,
    max_iterations: usize,
    flatness_threshold: T,
}

/// Parks-McClellan design parameters trait.
///
/// This trait defines the methods that the Parks-McClellan implementation,
/// [`pm_remez`](super::pm_remez), uses to define the type of filter that is
/// constructed and how the algorithm runs. It is implemented by the
/// [`PMParameters`] struct and by the object returned by the
/// [`pm_parameters`](super::pm_parameters) function, which provide the two
/// coding styles supported by this crate. If needed, the user can provide an
/// object that implements this trait to configure the Parks-McClellan algorithm
/// in a more flexible or specific way.
///
/// The type parameter `T` represents the scalar type to be used for the
/// computations. Typically it should implement [`Float`] and [`FloatConst`].
///
/// When an object implementing this trait is used with the
/// [`pm_remez`](super::pm_remez) function, the function may call the methods
/// defined by the trait at any time while the function runs. The `pm_remez`
/// function assumes that each of these methods always returns the same value
/// every time it is called during the execution of the function.
pub trait DesignParameters<T> {
    /// Returns the number of taps.
    ///
    /// This method indicates the number of taps of the FIR filter.
    fn num_taps(&self) -> usize;

    /// Returns the list of bands.
    ///
    /// This method gives a list of sub-bands of the interval [0.0, 0.5] in
    /// where the Parks-McClellan algorithm tries to minimize the maximum
    /// weighted error.
    fn bands(&self) -> &[Band<T>];

    /// Returns the symmetry of the FIR filter taps.
    ///
    /// This method indicates whether the FIR filter taps have even or odd
    /// symmetry.
    fn symmetry(&self) -> Symmetry;

    /// Returns the desired response.
    ///
    /// This function returns a closure that the Parks-McClellan algorithm
    /// evaluates at points belonging to the bands listed by the `bands`
    /// method. Due to numerical precision errors, the Parks-McClellan algorithm
    /// may also try to evaluate this closure at points which are outside all of
    /// these bands, but very close to one of the band endpoints. The closure
    /// should handle this situation gracefully, returning an output that is
    /// very close to that intended at the corresponding band endpoint.
    fn desired_response(&self) -> impl Fn(T) -> T;

    /// Returns the weight function.
    ///
    /// This function returns a closure that the Parks-McClellan algorithm
    /// evaluates at points belonging to the bands listed by the `bands`
    /// method. Due to numerical precision errors, the Parks-McClellan algorithm
    /// may also try to evaluate this closure at points which are outside all of
    /// these bands, but very close to one of the band endpoints. The closure
    /// should handle this situation gracefully, returning an output that is
    /// very close to that intended at the corresponding band endpoint.
    fn weights(&self) -> impl Fn(T) -> T;

    /// Returns the degree of the Chebyshev proxy that is used for Chebyshev
    /// proxy root finding.
    ///
    /// In order to find the local extrema of the weighted error function, the
    /// [`pm_remez`](super::pm_remez) function uses Chebyshev proxy root
    /// finding. In each interval delimited by the current extrema candidates,
    /// the Chebyshev interpolant of the weighted error function, with the
    /// degree specified by this function, is computed, and the roots of its
    /// derivative are found. These roots are used as candidates for local
    /// extrema.
    fn chebyshev_proxy_degree(&self) -> usize;

    /// Returns the maximum number of Remez exchange iterations to be performed
    /// by [`pm_remez`](super::pm_remez).
    fn max_iterations(&self) -> usize;

    /// Returns the flatness threshold to be used to stop the Remez exchange
    /// before the maximum number of iterations is reached.
    ///
    /// The flatness metric is used to assess if the algorithm has converged. It
    /// is defined as the difference between the maximum absolute value of the
    /// weighted error over all the extremal frequencies minus the minimum
    /// absolute value of the weighted error over all the extremal frequencies,
    /// divided by the maximum absolute value of the weighted error over all the
    /// extremal frequencies.
    ///
    /// If the current flatness is below the threshold defined by this function,
    /// the algorithm stops.
    fn flatness_threshold(&self) -> T;
}

impl<T, D, W> DesignParameters<T> for PMParameters<T, D, W>
where
    T: Copy,
    D: Fn(T) -> T,
    W: Fn(T) -> T,
{
    fn num_taps(&self) -> usize {
        self.num_taps
    }
    fn bands(&self) -> &[Band<T>] {
        &self.bands
    }
    fn symmetry(&self) -> Symmetry {
        self.symmetry
    }
    fn desired_response(&self) -> impl Fn(T) -> T {
        &self.desired_response
    }
    fn weights(&self) -> impl Fn(T) -> T {
        &self.weights
    }
    fn chebyshev_proxy_degree(&self) -> usize {
        self.cheby_proxy_m
    }
    fn max_iterations(&self) -> usize {
        self.max_iterations
    }
    fn flatness_threshold(&self) -> T {
        self.flatness_threshold
    }
}

/// Parks-McClellan design parameters setter trait.
///
/// This trait is implemented by [`PMParameters`] and allows the default values
/// to be modified by calling the methods defined by the trait. It is also
/// implemented by the object returned by
/// [`pm_parameters`](super::pm_parameters), so that this object can be used as
/// if it was a `PMParameters` object.
pub trait ParametersBuilder<T>: DesignParameters<T> {
    /// Sets the symmetry of the FIR filter taps.
    ///
    /// This method sets whether the FIR filter taps have even or odd
    /// symmetry.
    fn set_symmetry(&mut self, symmetry: Symmetry) -> &mut Self;

    /// Returns the degree of the Chebyshev proxy that is used for Chebyshev
    /// proxy root finding.
    ///
    /// See [`DesignParameters::chebyshev_proxy_degree`].
    fn set_chebyshev_proxy_degree(&mut self, degree: usize) -> &mut Self;

    /// Sets the maximum number of Remez exchange iterations to be performed
    /// by [`pm_remez`](super::pm_remez).
    fn set_max_iterations(&mut self, max_iterations: usize) -> &mut Self;

    /// Returns the flatness threshold to be used to stop the Remez exchange
    /// before the maximum number of iterations is reached.
    ///
    /// See [`DesignParameters::flatness_threshold`].
    fn set_flatness_threshold(&mut self, flatness_threshold: T) -> &mut Self;
}

impl<T: Float, D, W> PMParameters<T, D, W> {
    /// Creates new design parameters for the Parks-McClellan algorithm.
    ///
    /// The `num_taps` argument indicates the number of taps of the FIR
    /// filter. The `bands` argument is the list of sub-bands of the interval
    /// [0.0, 0.5] on which the maximum weighted error is minimized. The
    /// `desired_response` and `weight` arguments indicate the desired response
    /// and weight function of the filter. Typically these should be closures
    /// `Fn(T) -> T`.
    ///
    /// Other parameters required for the Parks-McClellan algorithm are given
    /// default values and can be changed using the methods defined by the
    /// [`ParametersBuilder`] trait. By default, the symmetry of the FIR filter
    /// taps is set to even, and the other parameters that can be changed by
    /// this trait are given reasonable default values.
    ///
    /// Due to numerical precision errors, the Parks-McClellan algorithm may try
    /// to evaluate `desired_response` and `weights` at points which are outside
    /// all of the bands specified in `bands`, but very close to one of the band
    /// endpoints. The `desired_response` and `weights` functions should handle
    /// this situation gracefully, and return an output that is very close to
    /// that intended at the corresponding band endpoint.
    pub fn new(
        num_taps: usize,
        bands: Vec<Band<T>>,
        desired_response: D,
        weights: W,
    ) -> Result<Self> {
        Ok(PMParameters {
            num_taps,
            bands,
            symmetry: Symmetry::Even,
            desired_response,
            weights,
            cheby_proxy_m: 8,
            max_iterations: 100,
            flatness_threshold: T::from(1e-3).unwrap(),
        })
    }
}

impl<T, D, W> ParametersBuilder<T> for PMParameters<T, D, W>
where
    T: Copy,
    D: Fn(T) -> T,
    W: Fn(T) -> T,
{
    fn set_symmetry(&mut self, symmetry: Symmetry) -> &mut Self {
        self.symmetry = symmetry;
        self
    }

    fn set_chebyshev_proxy_degree(&mut self, degree: usize) -> &mut Self {
        self.cheby_proxy_m = degree;
        self
    }

    fn set_max_iterations(&mut self, max_iterations: usize) -> &mut Self {
        self.max_iterations = max_iterations;
        self
    }

    fn set_flatness_threshold(&mut self, flatness_threshold: T) -> &mut Self {
        self.flatness_threshold = flatness_threshold;
        self
    }
}
