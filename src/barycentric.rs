use crate::ExtremaCandidate;
use num_traits::Float;

// Calculate barycentric weigths
//
// w_k = 1 / prod_{j != k} (x_j - x_k).
//
// There are two optimizations for numerical errors in this calculation:
//
// a) Instead of multiplying (x_j - x_k), multiply (x_j - x_k) * 2. This
// helps prevent the exponent of the product from getting too small. The
// lagrange interpolation formula is homogeneous on the barycentric weights,
// so this change is acceptable.
//
// b) Stride through the elements in the product. I don't understand why
// this helps, but other implementations do it.
fn compute_barycentric_weights_with_key<'a, A, F, T>(
    x: &'a [A],
    key: F,
) -> impl Iterator<Item = T> + 'a
where
    F: Fn(&A) -> T + 'a,
    T: Float + 'a,
{
    let stride = (x.len() - 2) / 15 + 1;
    let one = T::one();
    let two = T::from(2).unwrap();
    x.iter().enumerate().map(move |(k, xk)| {
        let mut prod = one;
        for a in 0..stride {
            for j in (a..x.len()).step_by(stride) {
                if j != k {
                    prod = prod * ((key(xk) - key(&x[j])) * two);
                }
            }
        }
        prod.recip()
    })
}

pub fn compute_barycentric_weights<T: Float>(x: &[T]) -> impl Iterator<Item = T> + '_ {
    compute_barycentric_weights_with_key(x, |x| *x)
}

// Calculate delta
//
// delta = sum_k w_k D(f_k) / sum_k (-1)^k w_k/W(f_k),
//
// where f_k are extremal frequencies and D and W are the desired response
// and weight.
pub fn compute_delta<T: Float>(wk: &[T], desired: &[T], weights: &[T]) -> T {
    let mut delta_numer = T::zero();
    let mut delta_denom = T::zero();
    for (k, ((&w, &des), &wei)) in wk
        .iter()
        .zip(desired.iter())
        .zip(weights.iter())
        .enumerate()
    {
        delta_numer = delta_numer + w * des;
        let z = w / wei;
        if k % 2 != 0 {
            delta_denom = delta_denom - z;
        } else {
            delta_denom = delta_denom + z;
        }
    }
    delta_numer / delta_denom
}

// Calculate delta from x.
//
// This is used to prune extrema candidates (but the code that calls it is
// currently disabled).
#[allow(dead_code)]
pub fn compute_delta_from_candidates<T: Float>(candidates: &[ExtremaCandidate<T>]) -> T {
    let wk = compute_barycentric_weights_with_key(candidates, |a| a.x);
    let mut delta_numer = T::zero();
    let mut delta_denom = T::zero();
    for (k, (w, a)) in wk.zip(candidates.iter()).enumerate() {
        delta_numer = delta_numer + w * a.desired;
        let z = w / a.weight;
        if k % 2 != 0 {
            delta_denom = delta_denom - z;
        } else {
            delta_denom = delta_denom + z;
        }
    }
    delta_numer / delta_denom
}

// Calculate y_k, the abscisas for Lagrange interpolation
//
// y_k = D(f_k) - (-1)^k delta / W(f_k)
pub fn compute_lagrange_abscisa<'a, T: Float>(
    delta: T,
    desired: &'a [T],
    weights: &'a [T],
) -> impl Iterator<Item = T> + 'a {
    desired
        .iter()
        .zip(weights.iter())
        .enumerate()
        .map(move |(k, (&des, &wei))| {
            let z = delta / wei;
            if k % 2 != 0 { des + z } else { des - z }
        })
}

// Compute H(arccos(x0))
pub fn compute_freq_response<T: Float>(x0: T, x: &[T], wk: &[T], yk: &[T]) -> T {
    let mut h_numer = T::zero();
    let mut h_denom = T::zero();
    for ((&xk, &w), &y) in x.iter().zip(wk.iter()).zip(yk.iter()) {
        if x0 == xk {
            // special case where we are evaluating at one of the
            // interpolation nodes
            return y;
        }
        let z = w / (x0 - xk);
        h_numer = h_numer + z * y;
        h_denom = h_denom + z;
    }
    h_numer / h_denom
}

fn compute_error_common<T, D, W>(
    x0: T,
    x: &[T],
    wk: &[T],
    yk: &[T],
    desired: D,
    weights: W,
) -> (T, T, T)
where
    T: Float,
    D: Fn(T) -> T,
    W: Fn(T) -> T,
{
    let h = compute_freq_response(x0, x, wk, yk);
    let f = x0.acos();
    let d = desired(f);
    let w = weights(f);
    let error = w * (d - h);
    (error, d, w)
}

// Compute E(f) = W(f) * (D(f) - H(f)), where cos(f) = x0
pub fn compute_error<T, D, W>(x0: T, x: &[T], wk: &[T], yk: &[T], desired: D, weights: W) -> T
where
    T: Float,
    D: Fn(T) -> T,
    W: Fn(T) -> T,
{
    compute_error_common(x0, x, wk, yk, desired, weights).0
}

pub fn compute_extrema_candidate<T, D, W>(
    x0: T,
    x: &[T],
    wk: &[T],
    yk: &[T],
    desired: D,
    weights: W,
) -> ExtremaCandidate<T>
where
    T: Float,
    D: Fn(T) -> T,
    W: Fn(T) -> T,
{
    let (error, d, w) = compute_error_common(x0, x, wk, yk, desired, weights);
    ExtremaCandidate {
        x: x0,
        error,
        desired: d,
        weight: w,
    }
}
