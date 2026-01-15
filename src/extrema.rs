use crate::barycentric::{compute_error, compute_extrema_candidate};
use crate::compute_cheby_coefficients;
use crate::eigenvalues::EigenvalueBackend;
use crate::error::{Error, Result};
use crate::types::Band;
use ndarray::Array2;
use num_traits::{Float, FloatConst};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Interval<T> {
    pub begin: T,
    pub end: T,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ExtremaCandidate<T> {
    pub x: T,
    pub error: T,
    pub desired: T,
    pub weight: T,
}

// Initial guess for extremal frequencies: evenly spaced over bands
pub fn initial_extremal_freqs<T: Float>(bands: &[Band<T>], num_functions: usize) -> Vec<T> {
    let mut total_band_length = T::zero();
    for band_length in bands.iter().map(|b| b.len()) {
        total_band_length = total_band_length + band_length;
    }
    let spacing = total_band_length / T::from(num_functions).unwrap();
    let mut consumed_length = T::zero();
    let num_bands = bands.len();
    let mut current_band = bands.iter().enumerate().peekable();
    (0..(num_functions + 1))
        .map(|j| {
            let s = T::from(j).unwrap() * spacing;
            debug_assert!(s >= consumed_length);
            let mut u = s - consumed_length;
            loop {
                let cband = current_band.peek().unwrap();
                let band_length = cband.1.len();
                // the second condition is to avoid going past the last band due to numerical rounding
                if u <= band_length || cband.0 == num_bands - 1 {
                    break;
                }
                current_band.next();
                consumed_length = consumed_length + band_length;
                u = s - consumed_length;
            }
            let cband = current_band.peek().unwrap();
            (cband.1.begin() + u).min(cband.1.end())
        })
        .collect()
}

// Compute subintervals containing the extremal points and band edges (in the
// [-1, 1] domain).
pub fn subdivide<T: Float>(x: &[T], bands_x: &[Interval<T>]) -> Vec<Interval<T>> {
    // reserve capacity for the worst case
    let mut subintervals = Vec::with_capacity(x.len() + bands_x.len());
    let mut xs = x.iter().rev().peekable();
    for band in bands_x {
        let mut begin = band.begin;
        loop {
            match xs.peek() {
                Some(&&a) => {
                    match a.partial_cmp(&band.end).unwrap() {
                        std::cmp::Ordering::Greater => {
                            // new point to the right of the band end: end
                            // interval at the right band edge, do not consume
                            // point.
                            subintervals.push(Interval {
                                begin,
                                end: band.end,
                            });
                            break;
                        }
                        std::cmp::Ordering::Equal => {
                            // new point exactly at the band end: end interval
                            // at the right band edge, consume point.
                            subintervals.push(Interval {
                                begin,
                                end: band.end,
                            });
                            xs.next();
                            break;
                        }
                        std::cmp::Ordering::Less => {
                            // new point inside the band: end interval at this
                            // point, consume point, the point is the begin of
                            // the next interval.
                            if begin != a {
                                subintervals.push(Interval { begin, end: a });
                                begin = a;
                            }
                            xs.next();
                        }
                    }
                }
                None => {
                    // no more points: end interval at the right band edge.
                    subintervals.push(Interval {
                        begin,
                        end: band.end,
                    });
                    break;
                }
            }
        }
    }
    // check that we have consumed all the points
    debug_assert!(xs.next().is_none());
    subintervals
}

// Find local extrema of error function in subinterval using Chebyshev proxy method
#[allow(clippy::too_many_arguments)]
pub fn find_extrema_in_subinterval<'a, T, D, W, B: EigenvalueBackend<T>>(
    interval: &Interval<T>,
    cheby_nodes: &[T],
    x: &'a [T],
    wk: &'a [T],
    yk: &'a [T],
    desired: D,
    weights: W,
    eigenvalue_backend: &B,
) -> Result<impl Iterator<Item = ExtremaCandidate<T>>>
where
    T: Float + FloatConst,
    D: Fn(T) -> T + 'a,
    W: Fn(T) -> T + 'a,
{
    // Compute Chebyshev proxy for error function in interval
    //
    // Scale Chebyshev nodes to interval and compute error function
    let mut cheby_nodes_errors: Vec<T> = {
        let scale = T::from(0.5).unwrap() * (interval.end - interval.begin);
        cheby_nodes
            .iter()
            .map(|&x0| {
                let cheby_node_scaled = (x0 + T::one()) * scale + interval.begin;
                compute_error(cheby_node_scaled, x, wk, yk, &desired, &weights)
            })
            .collect()
    };
    // Compute coefficients of first-order Chebyshev polynomial expansion
    let ak = compute_cheby_coefficients(&mut cheby_nodes_errors);

    // Compute derivative of Chebyshev proxy
    //
    // Compute coefficients of second-order Chevyshev polynomial expansion of
    // the derivative of the proxy.
    let mut ck: Vec<T> = ak
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &a)| T::from(k).unwrap() * a)
        .collect();

    // Remove high-order coefficients ck which are zero. The colleague matrix
    // definition needs the leading coefficient to be nonzero.
    let zero = T::zero();
    while *ck.last().unwrap() == zero {
        ck.pop();
        if ck.is_empty() {
            return Err(Error::ProxyDerivativeZero);
        }
    }

    // Compute colleague matrix of ck. Its eigenvalues are the zeros of the
    // derivative of the Chebyshev proxy.
    let s = ck.len() - 1;
    let mut colleague = Array2::<T>::zeros((s, s));
    let half = T::from(0.5).unwrap();
    for j in 0..s - 1 {
        colleague[(j, j + 1)] = half;
    }
    for j in 2..s {
        colleague[(j, j - 1)] = half;
    }
    let scale = T::from(-0.5).unwrap() / *ck.last().unwrap();
    for j in 0..s {
        let c = ck[s - 1 - j] * scale;
        colleague[(j, 0)] = if j == 1 { c + half } else { c };
    }
    // Balance matrix for better numerical conditioning
    balance_matrix(&mut colleague);

    // Compute eigenvalues of colleague matrix. These are the roots of the
    // derivative of the proxy.
    let eig = eigenvalue_backend.eigenvalues(colleague)?;

    // Filter only the roots that are real and inside [-1, 1]. Map them to
    // the original interval.
    let threshold = T::from(1e-20).unwrap();
    let limits = -T::one()..=T::one();
    let scale = T::from(0.5).unwrap() * (interval.end - interval.begin);
    let begin = interval.begin;
    Ok(eig.into_iter().filter_map(move |z| {
        if Float::abs(z.im) < threshold {
            let x0 = z.re;
            if limits.contains(&x0) {
                // map root to interval
                let y = (x0 + T::one()) * scale + begin;
                // evaluate error function
                Some(compute_extrema_candidate(y, x, wk, yk, &desired, &weights))
            } else {
                None
            }
        } else {
            None
        }
    }))
}

// Prune extrema candidates to leave only n of them. It assumes that the candidates are sorted.
pub fn prune_extrema_candidates<T: Float>(
    candidates: &[ExtremaCandidate<T>],
    n: usize,
) -> Result<Vec<ExtremaCandidate<T>>> {
    assert!(!candidates.is_empty());
    let mut pruned = Vec::with_capacity(candidates.len());
    let zero = T::zero();

    // From groups of adjacent extrema with the same sign, leave only the largest
    let mut b = candidates[0];
    let mut b_sign = b.error < zero;
    let mut b_abs = b.error.abs();
    for &a in candidates.iter().skip(1) {
        let a_sign = a.error < zero;
        let a_abs = a.error.abs();
        if a_sign != b_sign {
            pruned.push(b);
        }
        if a_sign != b_sign || a_abs > b_abs {
            b = a;
            b_sign = a_sign;
            b_abs = a_abs;
        }
    }
    pruned.push(b);

    if pruned.len() == n {
        return Ok(pruned);
    }
    if pruned.len() < n {
        return Err(Error::NotEnoughExtrema);
    }

    let to_remove = pruned.len() - n;
    // FIXME: This technique gives convergence problems in some cases, such as a
    // lowpass FIR with 1/f stopband response when the stopband weigth is set
    // large enough. The last extrema is always removed and the problem starts
    // ignoring the error at the end of the stopband.
    //
    // if to_remove == 1 {
    //     // Only one extrema needs to be removed. Consider the cases of removing
    //     // the first extrema and the last extrema, and compute delta for each of
    //     // them. Choose the option that gives a larger delta.
    //     let delta_keep_first = compute_delta_from_candidates(&pruned[..n]);
    //     let delta_keep_last = compute_delta_from_candidates(&pruned[1..]);
    //     if delta_keep_first >= delta_keep_last {
    //         // remove last candidate
    //         pruned.pop();
    //     } else {
    //         // remove first candidate
    //         pruned.remove(0);
    //     }
    //     return Ok(pruned);
    // }
    if to_remove % 2 == 1 {
        // An odd number of extrema need to be removed. Reduce this to the case
        // of an even number of extrema for removal by removing either the first
        // or last extrema, whichever has smaller error.
        if pruned[0].error.abs() >= pruned[pruned.len() - 1].error.abs() {
            pruned.pop();
        } else {
            pruned.remove(0);
        }
    }
    while pruned.len() > n {
        // An even number of extrema need to be removed. Find the pair of
        // elements that has smaller minimum absolute value among the two
        // elements of the pair and remove that pair.
        let idx = pruned
            .iter()
            .zip(pruned.iter().skip(1))
            .enumerate()
            .map(|(k, (a, b))| (k, a.error.abs().min(b.error.abs())))
            // unwrap will fail if there are NaN's
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        pruned.drain(idx..=idx + 1);
    }
    assert!(pruned.len() == n);
    Ok(pruned)
}

// Balance a matrix for eigenvalue calculation, as indicated in [5].
fn balance_matrix<T: Float>(a: &mut Array2<T>) {
    // Some constants to be used below
    let gamma = T::from(0.95).unwrap();
    let two = T::from(2.0).unwrap();
    let four = T::from(4.0).unwrap();
    let half = T::from(0.5).unwrap();
    let one = T::one();
    let zero = T::zero();

    // The algorithm in [5] has a preliminary step where rows and columns that
    // isolate an eignevalue (those that zero except on the diagonal element)
    // are pushed to the left or bottom of the matrix respectively. However, the
    // colleague matrix does not have any such rows or columns, so we don't need
    // this step.

    let n = a.nrows();
    let mut converged = false;
    while !converged {
        converged = true;
        for j in 0..n {
            let mut row_norm = zero;
            let mut col_norm = zero;
            for k in 0..n {
                // ignore the diagonal term, because the algorithm only works with
                // the off-diagonal matrix
                if k != j {
                    row_norm = row_norm + a[(j, k)].abs();
                    col_norm = col_norm + a[(k, j)].abs();
                }
            }
            if row_norm == zero || col_norm == zero {
                continue;
            }
            // Sum of original row norm and column norm. To be used in the
            // condition below.
            let norm_sum = row_norm + col_norm;
            // Implicitly finds the integer sigma such that
            // 2^{2*sigma - 1} < row_norm / col_norm <= 2^{2*sigma + 1}
            // and sets f = 2^sigma.
            let mut f = one;
            let row_norm_half = row_norm * half;
            // The is_normal serves to stop iteration if we run into numerical
            // trouble instead of looping forever.
            while col_norm.is_normal() && col_norm <= row_norm_half {
                f = f * two;
                col_norm = col_norm * four;
            }
            let row_norm_twice = row_norm * two;
            while col_norm.is_normal() && col_norm > row_norm_twice {
                f = f / two;
                col_norm = col_norm / four;
            }
            // By the end of these two loops col_norm has been replaced with
            // col_norm * f^2.

            // If we have run into trouble we just return
            if !col_norm.is_normal() {
                return;
            }

            // Check if
            // col_norm * f + row_norm / f < gamma * (col_norm + row_norm)
            // Since at this point col_norm contains col_norm * f^2, we multiply
            // both sides of the equation by f.
            if row_norm + col_norm < gamma * norm_sum * f {
                converged = false;
                let f_recip = f.recip();
                // Let D be a diagonal matrix that contains ones in all the
                // diagonal elements excepth the j-th, where it contains
                // f. Replace the matrix A by D^{-1}AD.
                for k in 0..n {
                    if k != j {
                        a[(j, k)] = a[(j, k)] * f_recip;
                        a[(k, j)] = a[(k, j)] * f;
                    }
                }
            }
        }
    }
}
