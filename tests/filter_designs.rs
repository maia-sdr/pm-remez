// needed because there are many unused things if there are no eigenvalue
// backends selected via features
#![cfg_attr(
    not(any(feature = "lapack-backend", feature = "faer-backend")),
    allow(dead_code)
)]

use num_traits::{Float, FloatConst, Zero};
use pm_remez::{
    BandSetting, Convf64, EigenvalueBackend, PMDesign, ParametersBuilder, Symmetry, constant,
    function, linear, pm_parameters, pm_remez_with_backend,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{f64::consts::PI, sync::Arc};

#[cfg(feature = "lapack-backend")]
use pm_remez::LapackBackend;

#[cfg(feature = "faer-backend")]
use pm_remez::FaerBackend;

#[cfg(feature = "nalgebra-backend")]
use pm_remez::NalgebraBackend;

#[cfg(feature = "num-bigfloat")]
use num_bigfloat::BigFloat;
#[cfg(feature = "num-bigfloat")]
use num_traits::One;

struct FirResponseCalculator {
    fft: Arc<dyn Fft<f64>>,
    buffer: Box<[Complex<f64>]>,
}

// The tests included here are inspired by the examples of the Python bindings
// documentation.

impl FirResponseCalculator {
    fn new(num_points: usize) -> FirResponseCalculator {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(2 * num_points);
        let buffer = vec![Complex::zero(); 2 * num_points].into_boxed_slice();
        FirResponseCalculator { fft, buffer }
    }

    fn frequencies<T: Float>(&self) -> Vec<T> {
        let scale = T::from(self.buffer.len()).unwrap().recip();
        (0..self.buffer.len() / 2)
            .map(|j| T::from(j).unwrap() * scale)
            .collect()
    }

    fn compute<T: Convf64>(&mut self, taps: &[T]) -> Vec<T> {
        assert!(taps.len() <= self.buffer.len());
        self.buffer.fill(Complex::zero());
        for (b, t) in self.buffer.iter_mut().zip(taps.iter()) {
            *b = t.to_f64().into();
        }
        self.fft.process(&mut self.buffer);
        self.buffer[..self.buffer.len() / 2]
            .iter()
            .map(|z| T::from_f64(z.norm()))
            .collect()
    }
}

fn design_antialias_lowpass<T, B>(
    decimation: usize,
    transition_bandwidth: T,
    numtaps: usize,
    stopband_weight: T,
    one_over_f: bool,
    eigenvalue_backend: &B,
) -> PMDesign<T>
where
    T: Float + FloatConst,
    B: EigenvalueBackend<T>,
{
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let decimation = T::from(decimation).unwrap();
    let passband_end = half * (one - transition_bandwidth) / decimation;
    let stopband_start = half * (one + transition_bandwidth) / decimation;
    let stopband_weight = if one_over_f {
        linear(stopband_weight, stopband_weight * half / stopband_start)
    } else {
        constant(stopband_weight)
    };
    let bands = [
        BandSetting::new(zero, passband_end, constant(one)).unwrap(),
        BandSetting::with_weight(stopband_start, half, constant(zero), stopband_weight).unwrap(),
    ];
    let params = pm_parameters(numtaps, &bands).unwrap();
    pm_remez_with_backend(&params, eigenvalue_backend).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check_antialias_lowpass_response<T: Float + Convf64>(
    response_calculator: &mut FirResponseCalculator,
    design: &PMDesign<T>,
    decimation: usize,
    transition_bandwidth: T,
    stopband_weight: T,
    one_over_f: bool,
    tolerance: T,
    end_skip: T,
) {
    let freqs: Vec<T> = response_calculator.frequencies();
    let response = response_calculator.compute(&design.impulse_response);
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let decimation = T::from(decimation).unwrap();
    let passband_end = half * (one - transition_bandwidth) / decimation;
    let stopband_start = half * (one + transition_bandwidth) / decimation;
    for (&f, &h) in freqs.iter().zip(response.iter()) {
        assert!(h <= (one + design.weighted_error) * (one + tolerance));
        if f <= passband_end {
            assert!((h - one).abs() <= design.weighted_error * (one + tolerance));
        } else if (stopband_start..=half - end_skip).contains(&f) {
            let weight = if one_over_f {
                stopband_weight * f / stopband_start
            } else {
                stopband_weight
            };
            assert!(h <= design.weighted_error / weight * (one + tolerance));
        }
    }
}

fn lowpass<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(2, 0.2f64, 35, 1.0, false, eigenvalue_backend);
    assert!(design.weighted_error < 6.8e-4);
    check_antialias_lowpass_response(
        &mut response_calculator,
        &design,
        2,
        0.2,
        1.0,
        false,
        tolerance,
        end_skip,
    );
}

#[cfg(feature = "lapack-backend")]
#[test]
fn lowpass_lapack() {
    lowpass(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn lowpass_faer() {
    lowpass(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn lowpass_nalgebra() {
    lowpass(&NalgebraBackend::default());
}

fn lowpass_stopband_weight<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(2, 0.2f64, 35, 10.0, false, eigenvalue_backend);
    assert!(design.weighted_error < 2.9e-3);
    check_antialias_lowpass_response(
        &mut response_calculator,
        &design,
        2,
        0.2,
        10.0,
        false,
        tolerance,
        end_skip,
    );
}

#[cfg(feature = "lapack-backend")]
#[test]
fn lowpass_stopband_weight_lapack() {
    lowpass_stopband_weight(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn lowpass_stopband_weight_faer() {
    lowpass_stopband_weight(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn lowpass_stopband_weight_nalgebra() {
    lowpass_stopband_weight(&NalgebraBackend::default());
}

fn lowpass_one_over_f<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(4, 0.2f64, 67, 10.0, true, eigenvalue_backend);
    assert!(design.weighted_error < 3.4e-3);
    check_antialias_lowpass_response(
        &mut response_calculator,
        &design,
        4,
        0.2,
        10.0,
        true,
        tolerance,
        end_skip,
    );
}

#[cfg(feature = "lapack-backend")]
#[test]
fn lowpass_one_over_f_lapack() {
    lowpass_one_over_f(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn lowpass_one_over_f_faer() {
    lowpass_one_over_f(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn lowpass_one_over_f_nalgebra() {
    lowpass_one_over_f(&NalgebraBackend::default());
}

#[cfg(feature = "num-bigfloat")]
fn lowpass_bigfloat<B: EigenvalueBackend<BigFloat>>(eigenvalue_backend: &B) {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(
        4,
        BigFloat::from(0.1),
        512,
        BigFloat::one(),
        true,
        eigenvalue_backend,
    );
    assert!(design.weighted_error < BigFloat::from(1.9e-10));
    check_antialias_lowpass_response(
        &mut response_calculator,
        &design,
        4,
        BigFloat::from(0.1),
        BigFloat::one(),
        true,
        BigFloat::from(tolerance),
        BigFloat::from(end_skip),
    );
}

#[cfg(all(feature = "num-bigfloat", feature = "lapack-backend"))]
#[test]
fn lowpass_bigfloat_lapack() {
    lowpass_bigfloat(&LapackBackend::default());
}

#[cfg(all(feature = "num-bigfloat", feature = "faer-backend"))]
#[test]
fn lowpass_bigfloat_faer() {
    lowpass_bigfloat(&FaerBackend::default());
}

#[cfg(all(feature = "num-bigfloat", feature = "nalgebra-backend"))]
#[test]
fn lowpass_bigfloat_nalgebra() {
    lowpass_bigfloat(&NalgebraBackend::default());
}

fn polyphase_filterbank<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let mut response_calculator = FirResponseCalculator::new(1 << 18);
    let tolerance = 1e-8;
    // ignore the response near Nyquist, since it regrows slightly, rather than
    // falling as 1/f.
    let end_skip = 2e-4;

    let num_channels = 2048;
    let taps_per_channel = 6;
    let design = design_antialias_lowpass(
        num_channels,
        0.35f64,
        num_channels * taps_per_channel,
        4.0,
        true,
        eigenvalue_backend,
    );
    assert!(design.weighted_error < 0.22);
    check_antialias_lowpass_response(
        &mut response_calculator,
        &design,
        num_channels,
        0.35,
        4.0,
        true,
        tolerance,
        end_skip,
    );
}

#[cfg(feature = "lapack-backend")]
#[test]
fn polyphase_filterbank_lapack() {
    polyphase_filterbank(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn polyphase_filterbank_faer() {
    polyphase_filterbank(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn polyphase_filterbank_nalgebra() {
    polyphase_filterbank(&NalgebraBackend::default());
}

fn bandpass<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let bands = [
        BandSetting::new(0.0, 0.075, constant(0.0)).unwrap(),
        BandSetting::new(0.1, 0.2, constant(1.0)).unwrap(),
        BandSetting::new(0.225, 0.275, constant(0.0)).unwrap(),
        BandSetting::new(0.3, 0.4, constant((10.0).recip().sqrt())).unwrap(),
        BandSetting::new(0.425, 0.5, constant(0.0)).unwrap(),
    ];
    let params = pm_parameters(135, &bands).unwrap();
    let design = pm_remez_with_backend(&params, eigenvalue_backend).unwrap();
    assert!(design.weighted_error < 9.4e-4);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    for (&f, &h) in response_calculator
        .frequencies()
        .iter()
        .zip(response_calculator.compute(&design.impulse_response).iter())
    {
        assert!(h <= (1.0 + design.weighted_error) * (1.0 + tolerance));
        if f <= 0.075 || (0.225..=0.275).contains(&f) || f >= 0.425 {
            assert!(h <= design.weighted_error * (1.0 + tolerance));
        } else if (0.1..=0.2).contains(&f) {
            assert!((h - 1.0).abs() <= design.weighted_error * (1.0 + tolerance));
        } else if (0.3..=0.4).contains(&f) {
            assert!((h - (10.0).recip().sqrt()).abs() <= design.weighted_error * (1.0 + tolerance));
        }
    }
}

#[cfg(feature = "lapack-backend")]
#[test]
fn bandpass_lapack() {
    bandpass(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn bandpass_faer() {
    bandpass(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn bandpass_nalgebra() {
    bandpass(&NalgebraBackend::default());
}

fn design_cic_compensation<B: EigenvalueBackend<f64>>(
    cic_stages: usize,
    cic_decimation: usize,
    decimation: usize,
    transition_bandwidth: f64,
    numtaps: usize,
    eigenvalue_backend: &B,
) -> PMDesign<f64> {
    let passband_end = 0.5 * (1.0 - transition_bandwidth) / decimation as f64;
    let stopband_start = 0.5 * (1.0 + transition_bandwidth) / decimation as f64;

    let desired = Box::new(move |f: f64| {
        if f < 1e-3 {
            // avoid math errors near f = 0
            1.0
        } else {
            let one_stage =
                cic_decimation as f64 * (PI * f / cic_decimation as f64).sin() / (PI * f).sin();
            one_stage.powi(cic_stages.try_into().unwrap())
        }
    });
    let bands = [
        BandSetting::new(0.0, passband_end, function(desired)).unwrap(),
        BandSetting::new(stopband_start, 0.5, constant(0.0)).unwrap(),
    ];
    let params = pm_parameters(numtaps, &bands).unwrap();
    pm_remez_with_backend(&params, eigenvalue_backend).unwrap()
}

fn check_cic_compensation_response(
    response_calculator: &mut FirResponseCalculator,
    design: &PMDesign<f64>,
    cic_stages: usize,
    cic_decimation: usize,
    decimation: usize,
    transition_bandwidth: f64,
    tolerance: f64,
) {
    let passband_end = 0.5 * (1.0 - transition_bandwidth) / decimation as f64;
    let stopband_start = 0.5 * (1.0 + transition_bandwidth) / decimation as f64;

    let freqs: Vec<f64> = response_calculator.frequencies();
    let response = response_calculator.compute(&design.impulse_response);
    for (&f, &h) in freqs.iter().zip(response.iter()) {
        if f <= passband_end {
            let desired = if f < 1e-3 {
                // avoid math errors near f = 0
                1.0
            } else {
                let one_stage =
                    cic_decimation as f64 * (PI * f / cic_decimation as f64).sin() / (PI * f).sin();
                one_stage.powi(cic_stages.try_into().unwrap())
            };
            assert!((h - desired).abs() <= design.weighted_error * (1.0 + tolerance));
        } else if f >= stopband_start {
            assert!(h <= design.weighted_error * (1.0 + tolerance));
        }
    }
}

fn cic_compensation<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let design = design_cic_compensation(4, 5, 3, 0.2, 53, eigenvalue_backend);
    assert!(design.weighted_error < 8.3e-4);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    check_cic_compensation_response(&mut response_calculator, &design, 4, 5, 3, 0.2, tolerance);
}

#[cfg(feature = "lapack-backend")]
#[test]
fn cic_compensation_lapack() {
    cic_compensation(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn cic_compensation_faer() {
    cic_compensation(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn cic_compensation_nalgebra() {
    cic_compensation(&NalgebraBackend::default());
}

fn design_hilbert<B: EigenvalueBackend<f64>>(
    transition_bandwidth: f64,
    numtaps: usize,
    eigenvalue_backend: &B,
) -> PMDesign<f64> {
    assert!(numtaps % 2 == 1);
    let allpass_begin = 0.25 * transition_bandwidth;
    let allpass_end = 0.5 - 0.25 * transition_bandwidth;
    let bands = [BandSetting::new(allpass_begin, allpass_end, constant(1.0)).unwrap()];
    let mut params = pm_parameters(numtaps, &bands).unwrap();
    params.set_symmetry(Symmetry::Odd);
    pm_remez_with_backend(&params, eigenvalue_backend).unwrap()
}

fn check_hilbert_response(
    response_calculator: &mut FirResponseCalculator,
    design: &PMDesign<f64>,
    transition_bandwidth: f64,
    tolerance: f64,
) {
    let allpass_begin = 0.25 * transition_bandwidth;
    let allpass_end = 0.5 - 0.25 * transition_bandwidth;
    let freqs: Vec<f64> = response_calculator.frequencies();
    let response = response_calculator.compute(&design.impulse_response);
    for (&f, &h) in freqs.iter().zip(response.iter()) {
        assert!(h <= (1.0 + design.weighted_error) * (1.0 + tolerance));
        if (allpass_begin..=allpass_end).contains(&f) {
            assert!((h - 1.0).abs() <= design.weighted_error * (1.0 + tolerance));
        }
    }
}

fn hilbert<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let design = design_hilbert(0.1, 43, eigenvalue_backend);
    assert!(design.weighted_error < 0.015);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    check_hilbert_response(&mut response_calculator, &design, 0.1, tolerance);
}

#[cfg(feature = "lapack-backend")]
#[test]
fn hilbert_lapack() {
    hilbert(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn hilbert_faer() {
    hilbert(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn hilbert_nalgebra() {
    hilbert(&NalgebraBackend::default());
}

fn design_differentiator<B: EigenvalueBackend<f64>>(
    transition_bandwidth: f64,
    numtaps: usize,
    eigenvalue_backend: &B,
) -> PMDesign<f64> {
    assert!(numtaps % 2 == 1);
    let allpass_end = 0.5 - 0.5 * transition_bandwidth;
    let bands = [BandSetting::new(0.0, allpass_end, linear(0.0, allpass_end)).unwrap()];
    let mut params = pm_parameters(numtaps, &bands).unwrap();
    params.set_symmetry(Symmetry::Odd);
    pm_remez_with_backend(&params, eigenvalue_backend).unwrap()
}

fn check_differentiator_response(
    response_calculator: &mut FirResponseCalculator,
    design: &PMDesign<f64>,
    transition_bandwidth: f64,
    tolerance: f64,
) {
    let allpass_end = 0.5 - 0.5 * transition_bandwidth;
    let freqs: Vec<f64> = response_calculator.frequencies();
    let response = response_calculator.compute(&design.impulse_response);
    for (&f, &h) in freqs.iter().zip(response.iter()) {
        assert!(h <= (1.0 + design.weighted_error) * (1.0 + tolerance));
        if f <= allpass_end {
            assert!((h - f).abs() <= design.weighted_error * (1.0 + tolerance));
        }
    }
}

fn differentiator<B: EigenvalueBackend<f64>>(eigenvalue_backend: &B) {
    let design = design_differentiator(0.1, 43, eigenvalue_backend);
    assert!(design.weighted_error < 2e-4);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-5;
    check_differentiator_response(&mut response_calculator, &design, 0.1, tolerance);
}

#[cfg(feature = "lapack-backend")]
#[test]
fn differentiator_lapack() {
    differentiator(&LapackBackend::default());
}

#[cfg(feature = "faer-backend")]
#[test]
fn differentiator_faer() {
    differentiator(&FaerBackend::default());
}

#[cfg(feature = "nalgebra-backend")]
#[test]
fn differentiator_nalgebra() {
    differentiator(&NalgebraBackend::default());
}
