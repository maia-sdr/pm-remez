//! FIR filter order estimates.
//!
//! This module contains functions to estimate the required order of an FIR
//! filter according to some design requirements.

use std::f64::consts::PI;

/// Estimates the required Parks-McClellan FIR length using the estimate from a
/// paper by Ichige etl.
///
/// This function uses the estimate presented in the following paper:
///
/// K. Ichige, M. Iwaki and R. Ishii, "Accurate estimation of minimum filter
/// length for optimum FIR digital filters," in IEEE Transactions on Circuits and
/// Systems II: Analog and Digital Signal Processing, vol. 47, no. 10,
/// pp. 1008-1016, Oct. 2000.
///
/// The parameters of this function are defined as follows: `fp` denotes the
/// passband edge frequency (normalized to a sample rate of 1), `delta_f`
/// denotes the transition bandwidth, which is defined as the difference between
/// the stopband edge and passband edge frequencies (both normalized to a sample
/// rate of 1), `delta_p` denotes the passband ripple, and `delta_s` denotes the
/// stopband ripple.
pub fn ichige(fp: f64, delta_f: f64, delta_p: f64, delta_s: f64) -> usize {
    let nc = (1.101 * (-(2.0 * delta_p).log10()).powf(1.1) / delta_f + 1.0).ceil();
    let v = 2.325 * (-(delta_p.log10())).powf(-0.445) * delta_f.powf(-1.39);
    let g = |x: f64| 2.0 / PI * (v * (x.recip() - (0.5 - delta_f).recip())).atan();
    let n3 = (nc * (g(fp) + g(0.5 - delta_f - fp) + 1.0) / 3.0).ceil();
    let nm = 0.52 * (delta_p / delta_s).log10() / delta_f * (-(delta_p.log10())).powf(0.17);
    let h =
        |x: f64, c: f64| 2.0 / PI * (c / delta_f * (x.recip() - (0.5 - delta_f).recip())).atan();
    let dn = (nm * (h(fp, 1.1) - (h(0.5 - delta_f - fp, 0.29) - 1.0) / 2.0)).ceil();
    (n3 + dn) as usize
}

#[cfg(test)]
mod test {
    #[test]
    fn ichige() {
        assert_eq!(super::ichige(0.1, 0.05, 0.01, 0.001), 54);
        assert_eq!(super::ichige(0.05, 0.05, 0.01, 0.001), 55);
        assert_eq!(super::ichige(0.025, 0.05, 0.01, 0.001), 57);
        assert_eq!(super::ichige(0.1, 0.1, 0.01, 0.001), 28);
        assert_eq!(super::ichige(0.01, 0.01, 0.01, 0.001), 271);
    }
}
