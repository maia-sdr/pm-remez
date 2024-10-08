//! # Parks-McClellan Remez FIR design algorithm
//!
//! The [`pm_remez`](crate) crate is a modern implementation of the
//! Parks-McClellan Remez exchange algorithm. It supports the design of FIR
//! filters with even symmetry and odd symmetry, and with an even number of taps
//! and an odd number of taps, by reducing all these cases to the even symmetry
//! odd number of taps case. The desired frequency response in each band, as
//! well as the weights, can be defined as arbitrary functions. The crate
//! supports using
//! [`num-bigfloat`](https://docs.rs/num-bigfloat/latest/num_bigfloat/), or any
//! other high precision floating point package that implements the [`Float`]
//! trait for the calculations. This can be used to solve numerically
//! challenging problems that are difficult to solve using `f64` arithmetic.
//!
//! The implementation draws ideas from \[2\] to make the algorithm robust
//! against numerical errors. These ideas include the use of Chebyshev proxy
//! root finding to find the extrema of the weighted error function in the Remez
//! exchange step.
//!
//! ## Examples
//!
//! The main function of this crate is [`pm_remez`], which takes a
//! [`DesignParameters`] object defining the filter to be constructed and
//! returns a [`PMDesign`] struct containing the filter taps and other
//! information. There are two coding styles in which this function can be used.
//!
//! The first style uses [`BandSetting`] to define each band, setting the
//! desired response and weight separately on each band. The following
//! constructs a lowpass filter by setting the desired response to a
//! [`constant`] of one in the passband and a constant of zero in the
//! stopband. Other kinds of reponses can be specified with [`linear`] (linear
//! slope interpolating two values) and [`function`] (arbitrary function). A
//! weight different from the default of `constant(1.0)` can be provided with
//! [`BandSetting::with_weight`]. The weight can be defined using `constant`,
//! `linear` or `function`.
//!
//! ```
//! # fn main() -> Result<(), pm_remez::error::Error> {
//! use pm_remez::{
//!     constant, pm_parameters, pm_remez,
//!     BandSetting, PMParameters, ParametersBuilder,
//! };
//! let bands = [
//!     BandSetting::new(0.0, 0.2, constant(1.0))?,
//!     BandSetting::new(0.3, 0.5, constant(0.0))?,
//! ];
//! let num_taps = 35;
//! let parameters = pm_parameters(num_taps, &bands)?;
//! let design = pm_remez(&parameters)?;
//! # Ok(())
//! # }
//! ```
//!
//! The second style is closer to how the `pm_remez` function is implemented. It
//! uses the [`PMParameters`] struct to define a list of [`Band`]s and the
//! desired response and weight as single functions for all the bands. The
//! `pm_remez` function is generic in the types of the desired response and
//! weight functions, so this style can enable more compile time optimizations
//! by using monomorphization.
//!
//! This designs the same lowpass filter using this second coding style.
//!
//! ```
//! # fn main() -> Result<(), pm_remez::error::Error> {
//! use pm_remez::{pm_remez, Band, BandSetting, PMParameters};
//! let num_taps = 35;
//! let parameters = PMParameters::new(
//!     num_taps,
//!     vec![Band::new(0.0, 0.2)?, Band::new(0.3, 0.5)?],
//!     |f| if f < 0.25 { 1.0 } else { 0.0 },
//!     |_| 1.0,
//! )?;
//! let design = pm_remez(&parameters)?;
//! # Ok(())
//! # }
//! ```
//!
//! The [documentation of the Python bindings](https://pm-remez.readthedocs.io/)
//! contains a series of examples that show how to use pm-remez to design
//! commonly used types of FIR filters. These examples are also useful to
//! understand how the Rust `pm_remez` function can be used.
//!
//! ## Building
//!
//! The `pm_remez` crate uses [`ndarray_linalg`] to solve eigenvalue problems.
//! This in turn depends on LAPACK. The `pm_remez` crate has several feature
//! flags that are used to select the LAPACK backend. Exactly one of these
//! features needs to be enabled to build `pm_remez`. The feature flags are
//! `openblas-static`, `openblas-system`, `netlib-static`, `netlib-system`,
//! `intel-mkl-static` and `intel-mkl-system`. The `-static` versions of each
//! flag build the LAPACK backend and link statically against it. The `-system`
//! versions link against a system-installed library (linking can be dynamic or
//! static depending on which type of library is installed).
//!
//! ## References
//!
//! \[1\] M. Ahsan and T. Saramaki, "Two Novel Implementations of the Remez
//! Multiple Exchange Algorithm for Optimum FIR Filter Design", MATLAB - A
//! Fundamental Tool for Scientific Computing and Engineering Applications -
//! Volume 2. InTech, Sep. 26, 2012.
//!
//! \[2\] S.I. Filip. "A Robust and Scalable Implementation of the
//! Parks-McClellan Algorithm for Designing FIR Filters," in ACM
//! Trans. Math. Softw. 43, 1, Article 7, March 2017.
//!
//! \[3\] J. McClellan, T. Parks and L. Rabiner, "A computer program for designing
//! optimum FIR linear phase digital filters," in IEEE Transactions on Audio and
//! Electroacoustics, vol. 21, no. 6, pp. 506-526, December 1973
//!
//! \[4\] T. Parks and J. McClellan, "Chebyshev Approximation for Nonrecursive
//! Digital Filters with Linear Phase," in IEEE Transactions on Circuit Theory,
//! vol. 19, no. 2, pp. 189-194, March 1972.
//!
//! \[5\] B.N. Parlett and C. Reinsch, "Balancing a matrix for calculation of
//! eigenvalues and eigenvectors". Numer. Math. 13, 293–304 (1969).
//!

#![warn(missing_docs)]

use itertools::{Itertools, MinMaxResult};
use num_traits::{Float, FloatConst};

mod bands;
use bands::*;
mod barycentric;
use barycentric::*;
mod chebyshev;
use chebyshev::compute_cheby_coefficients;
pub mod error;
use error::Result;
mod extrema;
use extrema::*;
mod fir_types;
use fir_types::*;
mod lapack;
pub use lapack::{IsLapack, ToLapack};
pub mod order_estimates;
#[cfg(feature = "python")]
mod python;
mod requirements;
pub use requirements::{constant, function, linear, pm_parameters, BandSetting, Setting};
mod types;
pub use types::{Band, DesignParameters, PMDesign, PMParameters, ParametersBuilder, Symmetry};

/// Parks-McClellan Remez exchange algorithm.
///
/// This function runs the Parks-McClellan Remez exchange algorithm to try to
/// find the optimal FIR filter that minimizes the maximum weighted error in
/// some sub-bands of the interval [0.0, 0.5], according to the configuration
/// parameters given in the `parameters` argument.
///
/// The type parameter `T` represents the scalar used internally in all the
/// computations (except when using LAPACK to solve an eigenvalue problem, for
/// which the [`ToLapack`] implementation of `T` is used to convert `T` to a
/// LAPACK-compatible scalar type).
///
/// The type parameter `P` represents the type of the Parks-McClellan design
/// parameters. It needs to implement the [`DesignParameters`] trait. The
/// `pm_remez` function uses the methods defined by this trait to obtain the
/// parameters that it needs.
///
/// # Examples
///
/// See the [crate-level examples](crate#examples) for examples about how to use
/// this function in each of the two coding styles provided by this crate.
pub fn pm_remez<T, P>(parameters: &P) -> Result<PMDesign<T>>
where
    T: Float + FloatConst + ToLapack,
    P: DesignParameters<T>,
{
    let bands = parameters.bands();
    check_bands(bands)?;
    let mut bands = sort_bands(bands);
    let num_taps = parameters.num_taps();
    let odd_length = num_taps % 2 != 0;
    // Check that the frequency response is realizable by the requested FIR type.
    let desired_response = parameters.desired_response();
    let symmetry = parameters.symmetry();
    check_response(&bands, &desired_response, symmetry, odd_length)?;
    // Adjust bands to avoid singularities.
    adjust_bands(&mut bands, symmetry, odd_length);
    // Convert bands from cycles/sample to radians/sample.
    for band in bands.iter_mut() {
        band.convert_to_radians();
    }

    // Adjust desired response and weights depending on FIR type. See Fig. 2 in
    // [3]. This also converts the argument of these functions from
    // cycles/sample to rad/sample.
    let desired = adjust_desired(desired_response, symmetry, odd_length);
    let weights = parameters.weights();
    let weights = adjust_weights(weights, symmetry, odd_length);

    // Number of cosine functions to use in approximation (n in [3]).
    let num_functions = match (symmetry, odd_length) {
        (Symmetry::Even, true) => num_taps / 2 + 1,
        _ => num_taps / 2,
    };

    // Calculate initial parameters

    let mut extremal_freqs = initial_extremal_freqs(&bands, num_functions);
    // x = cos(f), where f are the extremal freqs
    let mut x: Vec<T> = extremal_freqs.iter().map(|f| f.cos()).collect();
    let mut wk: Vec<T> = compute_barycentric_weights(&x).collect();
    let mut desired_x: Vec<T> = extremal_freqs.iter().map(|&f| desired(f)).collect();
    let mut weights_x: Vec<T> = extremal_freqs.iter().map(|&f| weights(f)).collect();
    let mut delta = compute_delta(&wk, &desired_x, &weights_x);
    let mut yk: Vec<T> = compute_lagrange_abscisa(delta, &desired_x, &weights_x).collect();
    let mut num_iterations = 0;
    let mut flatness = T::zero();
    let max_iterations = parameters.max_iterations();
    let cheby_proxy_m = parameters.chebyshev_proxy_degree();
    let flatness_threshold = parameters.flatness_threshold();
    for num_iter in 1..=max_iterations {
        num_iterations = num_iter;
        // Perform Remez exchange

        // Convert bands edges using x = cos(f). Note that cos() is decreasing, so
        // we use rev() and swap end and begin to obtain an output in increasing
        // order.
        let bands_x: Vec<Interval<T>> = bands
            .iter()
            .rev()
            .map(|b| Interval {
                begin: b.end().cos(),
                end: b.begin().cos(),
            })
            .collect();
        let subintervals = subdivide(&x, &bands_x);
        // TODO: use with_capacity with an upper estimate
        let mut remez_candidates: Vec<ExtremaCandidate<T>> = Vec::new();

        // Add subinterval endpoints to the candidate list
        remez_candidates.extend(subintervals.iter().flat_map(|interval| {
            [
                compute_extrema_candidate(interval.begin, &x, &wk, &yk, &desired, &weights),
                compute_extrema_candidate(interval.end, &x, &wk, &yk, &desired, &weights),
            ]
            .into_iter()
        }));

        // Compute Chebyshev nodes for [-1, 1] interval
        let cheby_nodes: Vec<T> = {
            let scale = T::PI() / T::from(cheby_proxy_m).unwrap();
            (0..=cheby_proxy_m)
                .map(|j| (T::from(j).unwrap() * scale).cos())
                .collect()
        };
        // Add local extrema inside each subinterval to the candidate list
        for interval in &subintervals {
            remez_candidates.extend(find_extrema_in_subinterval(
                interval,
                &cheby_nodes,
                &x,
                &wk,
                &yk,
                &desired,
                &weights,
            )?);
        }

        // Sort candidates
        // unwrap will fail if there are NaN's in the x values
        remez_candidates.sort_unstable_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

        // Prune extrema candidates to leave only num_functions + 1 of them
        let remez_candidates = prune_extrema_candidates(&remez_candidates, num_functions + 1)?;

        // Find largest and smallest error value in extrema candidates to assess convergence
        let MinMaxResult::MinMax(min_error, max_error) =
            remez_candidates.iter().map(|a| a.error.abs()).minmax()
        else {
            panic!("remez_candidates has two few elements to obtain minmax()")
        };
        flatness = (max_error - min_error) / max_error;

        // Set new extremal frequencies from candidates
        for ((f, x0), candidate) in extremal_freqs
            .iter_mut()
            .zip(x.iter_mut())
            // rev is used because acos() is a decreasing function
            .zip(remez_candidates.iter().rev())
        {
            *x0 = candidate.x;
            *f = candidate.x.acos();
        }
        // Compute new barycentric weights
        for (dst, src) in wk.iter_mut().zip(compute_barycentric_weights(&x)) {
            *dst = src;
        }
        // Compute new desired and weights
        for (des, &f) in desired_x.iter_mut().zip(extremal_freqs.iter()) {
            *des = desired(f);
        }
        for (wei, &f) in weights_x.iter_mut().zip(extremal_freqs.iter()) {
            *wei = weights(f);
        }
        // Compute new delta
        delta = compute_delta(&wk, &desired_x, &weights_x);
        // Compute new y_k
        for (dst, src) in yk
            .iter_mut()
            .zip(compute_lagrange_abscisa(delta, &desired_x, &weights_x))
        {
            *dst = src
        }

        if flatness <= flatness_threshold {
            // Convergence reached
            break;
        }
    }

    // Obtain the time-domain coefficients.
    //
    // This can be done by evaluating H(f) at the Chebyshev nodes of the second
    // kind, f = cos(k*pi/n), where H(f) = \sum_{0 <= k <= n} a_k * cos(k*f),
    // and then computing a_k as the coefficients in the expansion of H(cos(x))
    // in terms of Chebyshev polynomials of the first kind.
    let mut ck: Vec<T> = {
        let scale = T::PI() / T::from(num_functions - 1).unwrap();
        (0..num_functions)
            .map(|j| compute_freq_response((T::from(j).unwrap() * scale).cos(), &x, &wk, &yk))
            .collect()
    };
    let ak = compute_cheby_coefficients(&mut ck);

    // Convert extremal frequencies from radians/sample to cycles/sample.
    for f in extremal_freqs.iter_mut() {
        *f = *f / T::TAU();
    }

    Ok(PMDesign {
        impulse_response: h_from_ak(&ak, num_taps, symmetry, odd_length),
        weighted_error: delta.abs(),
        extremal_freqs,
        num_iterations,
        flatness,
    })
}
