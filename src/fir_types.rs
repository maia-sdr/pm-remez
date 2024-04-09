use crate::{
    error::{InvalidResponse, Result},
    types::{Band, Symmetry},
};
use num_traits::{Float, FloatConst};

// Adjust desired response depending on FIR type. See Fig. 2 in [3].
pub fn adjust_desired<'a, T, D>(
    desired: D,
    symmetry: Symmetry,
    odd_length: bool,
) -> impl Fn(T) -> T + 'a
where
    T: Float + FloatConst,
    D: Fn(T) -> T + 'a,
{
    // the frequencies f are in rad/sample, but desired is in cycles/sample
    move |f| {
        let d = (desired)(f / T::TAU());
        match (symmetry, odd_length) {
            (Symmetry::Even, true) => d,
            (Symmetry::Even, false) => d / (T::from(0.5).unwrap() * f).cos(),
            (Symmetry::Odd, false) => d / (T::from(0.5).unwrap() * f).sin(),
            (Symmetry::Odd, true) => d / f.sin(),
        }
    }
}

// Adjust weights depending on FIR type. See Fig. 2 in [3]
pub fn adjust_weights<'a, T, W>(
    weights: W,
    symmetry: Symmetry,
    odd_length: bool,
) -> impl Fn(T) -> T + 'a
where
    T: Float + FloatConst,
    W: Fn(T) -> T + 'a,
{
    // the frequencies f are in rad/sample, but weights is in cycles/sample
    move |f| {
        let weight = (weights)(f / T::TAU());
        match (symmetry, odd_length) {
            (Symmetry::Even, true) => weight,
            (Symmetry::Even, false) => weight * (T::from(0.5).unwrap() * f).cos(),
            (Symmetry::Odd, false) => weight * (T::from(0.5).unwrap() * f).sin(),
            (Symmetry::Odd, true) => weight * f.sin(),
        }
    }
}

// Check that the desired response is realizable by the requested FIR type.
//
// This function must be called before calling adjust_bands, because it only
// checks the potentially problematic frequencies f = 0, f = 0.5 (at this point
// the frequencies are still given in cycles/sample). It assumes that the
// desired response is continuous around these frequencies if they are included
// in one of the bands.
pub fn check_response<T, D>(
    bands: &[Band<T>],
    desired: D,
    symmetry: Symmetry,
    odd_length: bool,
) -> Result<()>
where
    T: Float,
    D: Fn(T) -> T,
{
    let zero = T::zero();
    let nyq = T::from(0.5).unwrap();
    match (symmetry, odd_length) {
        (Symmetry::Even, true) => {
            // there are no singular points
            Ok(())
        }
        (Symmetry::Even, false) => {
            // f = 0.5 is a singular point
            if bands.last().unwrap().end() == nyq && desired(nyq) != zero {
                Err(InvalidResponse::EvenSymmEvenLengthNyquist)?
            } else {
                Ok(())
            }
        }
        (Symmetry::Odd, false) => {
            // f = 0 is a singular point
            if bands[0].begin() == zero && desired(zero) != zero {
                Err(InvalidResponse::OddSymmEvenLengthDC)?
            } else {
                Ok(())
            }
        }
        (Symmetry::Odd, true) => {
            // f = 0, f = pi are singular points
            if bands[0].begin() == zero && desired(zero) != zero {
                Err(InvalidResponse::OddSymmOddLengthDC)?
            } else if bands.last().unwrap().end() == nyq && desired(nyq) != zero {
                Err(InvalidResponse::OddSymmOddLengthNyquist)?
            } else {
                Ok(())
            }
        }
    }
}

// Adjust band edges to avoid singularities in adjusted response.
//
// When this function is called, the band edges are still in cycles/sample.
pub fn adjust_bands<T: Float>(bands: &mut Vec<Band<T>>, symmetry: Symmetry, odd_length: bool) {
    fn avoid_zero<T: Float>(bands: &mut Vec<Band<T>>) {
        let eps = T::from(1e-4).unwrap();
        let zero = T::zero();
        if bands[0].begin() == zero {
            let replacement = eps;
            if bands[0].end() < replacement {
                // remove band to avoid empty band
                bands.remove(0);
            } else {
                // replace band
                bands[0] = Band::new(replacement, bands[0].end()).unwrap();
            }
        }
    }

    fn avoid_nyquist<T: Float>(bands: &mut Vec<Band<T>>) {
        let eps = T::from(1e-4).unwrap();
        let nyq = T::from(0.5).unwrap();
        if bands.last().unwrap().end() == nyq {
            let replacement = nyq - eps;
            if bands.last().unwrap().begin() > replacement {
                // remove band to avoid empty band
                bands.pop();
            } else {
                // replace band
                *bands.last_mut().unwrap() =
                    Band::new(bands.last().unwrap().begin(), replacement).unwrap();
            }
        }
    }

    match (symmetry, odd_length) {
        (Symmetry::Even, true) => {
            // nothing to do in this case, since there are no singularities
        }
        (Symmetry::Even, false) => {
            // there is a singularity at f = 0.5
            avoid_nyquist(bands);
        }
        (Symmetry::Odd, false) => {
            // there is a singularity at f = 0
            avoid_zero(bands);
        }
        (Symmetry::Odd, true) => {
            // there are singularities at f = 0, f = 0.5
            avoid_zero(bands);
            avoid_nyquist(bands);
        }
    }
}

// Obtain impulse response from coefficients a_k of the expression
//
// H(f) = sum_k a_k cos(f)
//
// See equations (3) - (12) in [3].
pub fn h_from_ak<T: Float>(
    ak: &[T],
    num_taps: usize,
    symmetry: Symmetry,
    odd_length: bool,
) -> Vec<T> {
    let mut h = Vec::with_capacity(num_taps);
    let scale = T::from(0.5).unwrap();
    let scale4 = T::from(0.25).unwrap();
    match (symmetry, odd_length) {
        (Symmetry::Even, true) => {
            h.extend(ak[1..].iter().rev().map(|&a| a * scale));
            h.push(ak[0]);
            h.extend(ak[1..].iter().map(|&a| a * scale));
        }
        (Symmetry::Even, false) => {
            h.push(*ak.last().unwrap() * scale4);
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(2))
                    .rev()
                    .map(|(&b1, &b2)| scale4 * (b1 + b2)),
            );
            h.push(ak[0] * scale + ak[1] * scale4);
            h.push(ak[0] * scale + ak[1] * scale4);
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(2))
                    .rev()
                    .map(|(&b1, &b2)| scale4 * (b1 + b2))
                    .rev(),
            );
            h.push(*ak.last().unwrap() * scale4);
        }
        (Symmetry::Odd, true) => {
            h.push(*ak.last().unwrap() * scale4);
            h.push(ak[ak.len() - 2] * scale4);
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(3))
                    .rev()
                    .map(|(&c1, &c2)| scale4 * (c1 - c2)),
            );
            h.push(ak[0] * scale - ak[2] * scale4);
            h.push(T::zero());
            h.push(-(ak[0] * scale - ak[2] * scale4));
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(3))
                    .rev()
                    .map(|(&c1, &c2)| -(scale4 * (c1 - c2)))
                    .rev(),
            );
            h.push(-(ak[ak.len() - 2] * scale4));
            h.push(-(*ak.last().unwrap() * scale4));
        }
        (Symmetry::Odd, false) => {
            h.push(*ak.last().unwrap() * scale4);
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(2))
                    .rev()
                    .map(|(&d1, &d2)| scale4 * (d1 - d2)),
            );
            h.push(ak[0] * scale - ak[1] * scale4);
            h.push(-(ak[0] * scale - ak[1] * scale4));
            h.extend(
                ak.iter()
                    .skip(1)
                    .zip(ak.iter().skip(2))
                    .rev()
                    .map(|(&d1, &d2)| -(scale4 * (d1 - d2)))
                    .rev(),
            );
            h.push(-(*ak.last().unwrap() * scale4));
        }
    };
    debug_assert!(h.len() == num_taps);
    h
}
