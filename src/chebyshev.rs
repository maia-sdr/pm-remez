use num_traits::{Float, FloatConst};

// Compute Chebychev coefficients a_k of
//
// p(x) = \sum_{0 <= k <= n} a_k T_k(x),
//
// where T_k(x) = cos(k*arccos(x)).
//
// given the values of p at Chebyshev nodes of the second kind,
// that is, p(k*pi/n), 0 <= k <= n.
//
// The way to do this is as follows:
// Put b_0 = 2*a_0, b_1 = a_1, ..., b_{n-1} = a_{n-1}, b_n = 2*a_n.
// Then {c_k = p(k*pi/n)} is the DCT-I of {b_k}. Put
// d_0 = c_0 / 2, d_1 = c_1, ..., d_{n-1} = c_{n-1}, d_n = c_n / 2.
// Then,
//   b_k = 2/n sum_{0 <= j <= n} d_j cos(jk pi/n) =
//       = 2/n sum_{0 <= j <= n} d_j T_j(cos(k pi/n)).
// Thus, the coefficients b_k can be evaluated using Clenshaw's algorithm
// for the Chebyshev polynomials of the first kind T_j. From this, a_k
// are calculated.
//
// Note: this function overwrites p_values with temporary calculations.
pub fn compute_cheby_coefficients<T: Float + FloatConst>(p_values: &mut [T]) -> Vec<T> {
    // Step 2: divide ck[0] and ck[-1] by 2 to transform c_k into d_k.
    halve_first_and_last(p_values);

    // Step 3: use Clenshaw's algorithm to compute b_k.
    let n = p_values.len() - 1;
    let scale = T::from(2).unwrap() / T::from(n).unwrap();
    let mut bk: Vec<T> = (0..=n)
        .map(|j| {
            let f = T::from(j).unwrap() * T::PI() / T::from(n).unwrap();
            let x = f.cos();
            clenshaw(p_values, x) * scale
        })
        .collect();

    // Step 4: divide bk[0] and bk[-1] by 2 to transform b_k into a_k.
    halve_first_and_last(&mut bk);
    bk
}

// Clenshaw's algorithm for Chebyshev polynomials of the first kind T_k.
// Evaluates \sum a_k T_k(x).
fn clenshaw<T: Float>(a: &[T], x: T) -> T {
    assert!(a.len() >= 2);
    let mut b2 = T::zero();
    let mut b1 = *a.last().unwrap();
    let two_x = T::from(2).unwrap() * x;
    for &ak in a[1..a.len() - 1].iter().rev() {
        let tmp = two_x * b1 - b2 + ak;
        b2 = b1;
        b1 = tmp;
    }
    x * b1 - b2 + a[0]
}

// Divides the first and last coefficient by 2
fn halve_first_and_last<T: Float>(a: &mut [T]) {
    assert!(a.len() >= 2);
    let scale = T::from(0.5).unwrap();
    let x = a.first_mut().unwrap();
    *x = *x * scale;
    let x = a.last_mut().unwrap();
    *x = *x * scale;
}

// Compute Chebyshev nodes of the second kind for the [-1, 1] interval
// (this returns n + 1 nodes)
pub fn chebyshev_nodes<T: Float + FloatConst>(n: usize) -> impl Iterator<Item = T> {
    let scale = T::PI() / T::from(n).unwrap();
    (0..=n).map(move |j| (T::from(j).unwrap() * scale).cos())
}
