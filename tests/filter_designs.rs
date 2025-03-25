use num_traits::{Float, FloatConst, Zero};
use pm_remez::{
    BandSetting, PMDesign, ParametersBuilder, Symmetry, ToLapack, constant, function, linear,
    pm_parameters, pm_remez,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{f64::consts::PI, sync::Arc};

#[cfg(feature = "num-bigfloat")]
use num_bigfloat::BigFloat;

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

    fn compute<T: ToLapack<Lapack = f64>>(&mut self, taps: &[T]) -> Vec<T> {
        assert!(taps.len() <= self.buffer.len());
        self.buffer.fill(Complex::zero());
        for (b, t) in self.buffer.iter_mut().zip(taps.iter()) {
            *b = t.to_lapack().into();
        }
        self.fft.process(&mut self.buffer);
        self.buffer[..self.buffer.len() / 2]
            .iter()
            .map(|z| T::from_lapack(&z.norm()))
            .collect()
    }
}

fn design_antialias_lowpass<T: Float + FloatConst + ToLapack>(
    decimation: usize,
    transition_bandwidth: T,
    numtaps: usize,
    stopband_weight: T,
    one_over_f: bool,
) -> PMDesign<T> {
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
    pm_remez(&params).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check_antialias_lowpass_response<T: Float + ToLapack<Lapack = f64>>(
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

#[test]
fn lowpass() {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(2, 0.2f64, 35, 1.0, false);
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

#[test]
fn lowpass_stopband_weight() {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(2, 0.2f64, 35, 10.0, false);
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

#[test]
fn lowpass_one_over_f() {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(4, 0.2f64, 67, 10.0, true);
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

#[test]
#[cfg(feature = "num-bigfloat")]
fn lowpass_bigfloat() {
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    let end_skip = 0.0;
    let design = design_antialias_lowpass(4, BigFloat::from(0.1), 512, BigFloat::one(), true);
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

#[test]
fn polyphase_filterbank() {
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

#[test]
fn bandpass() {
    let bands = [
        BandSetting::new(0.0, 0.075, constant(0.0)).unwrap(),
        BandSetting::new(0.1, 0.2, constant(1.0)).unwrap(),
        BandSetting::new(0.225, 0.275, constant(0.0)).unwrap(),
        BandSetting::new(0.3, 0.4, constant((10.0).recip().sqrt())).unwrap(),
        BandSetting::new(0.425, 0.5, constant(0.0)).unwrap(),
    ];
    let params = pm_parameters(135, &bands).unwrap();
    let design = pm_remez(&params).unwrap();
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

fn design_cic_compensation(
    cic_stages: usize,
    cic_decimation: usize,
    decimation: usize,
    transition_bandwidth: f64,
    numtaps: usize,
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
    pm_remez(&params).unwrap()
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

#[test]
fn cic_compensation() {
    let design = design_cic_compensation(4, 5, 3, 0.2, 53);
    assert!(design.weighted_error < 8.3e-4);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    check_cic_compensation_response(&mut response_calculator, &design, 4, 5, 3, 0.2, tolerance);
}

fn design_hilbert(transition_bandwidth: f64, numtaps: usize) -> PMDesign<f64> {
    assert!(numtaps % 2 == 1);
    let allpass_begin = 0.25 * transition_bandwidth;
    let allpass_end = 0.5 - 0.25 * transition_bandwidth;
    let bands = [BandSetting::new(allpass_begin, allpass_end, constant(1.0)).unwrap()];
    let mut params = pm_parameters(numtaps, &bands).unwrap();
    params.set_symmetry(Symmetry::Odd);
    pm_remez(&params).unwrap()
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

#[test]
fn hilbert() {
    let design = design_hilbert(0.1, 43);
    assert!(design.weighted_error < 0.015);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-10;
    check_hilbert_response(&mut response_calculator, &design, 0.1, tolerance);
}

fn design_differentiator(transition_bandwidth: f64, numtaps: usize) -> PMDesign<f64> {
    assert!(numtaps % 2 == 1);
    let allpass_end = 0.5 - 0.5 * transition_bandwidth;
    let bands = [BandSetting::new(0.0, allpass_end, linear(0.0, allpass_end)).unwrap()];
    let mut params = pm_parameters(numtaps, &bands).unwrap();
    params.set_symmetry(Symmetry::Odd);
    pm_remez(&params).unwrap()
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

#[test]
fn differentiator() {
    let design = design_differentiator(0.1, 43);
    assert!(design.weighted_error < 2e-4);
    let mut response_calculator = FirResponseCalculator::new(4096);
    let tolerance = 1e-5;
    check_differentiator_response(&mut response_calculator, &design, 0.1, tolerance);
}
