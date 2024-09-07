# pm-remez: Parks-McClellan Remez FIR design algorithm

[![Crates.io][crates-badge]][crates-url]
[![Rust](https://github.com/maia-sdr/pm-remez/actions/workflows/rust.yml/badge.svg)](https://github.com/maia-sdr/pm-remez/actions/workflows/rust.yml)
[![Rust Docs][docs-badge]][docs-url]
[![Python](https://github.com/maia-sdr/pm-remez/actions/workflows/maturin.yml/badge.svg)](https://github.com/maia-sdr/pm-remez/actions/workflows/maturin.yml)
[![Python Docs](https://readthedocs.org/projects/pm-remez/badge/?version=latest)](https://pm-remez.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[crates-badge]: https://img.shields.io/crates/v/pm-remez
[crates-url]: https://crates.io/crates/pm-remez
[docs-badge]: https://docs.rs/pm-remez/badge.svg
[docs-url]: https://docs.rs/pm-remez

pm-remez is a modern Rust implementation of the Parks-McClellan Remez exchange
algorithm. It can be used as a Rust library and as a Python package via its
Python bindings.

pm-remez supports the design of FIR filters with even symmetry and odd symmetry,
and with an even number of taps and an odd number of taps, by reducing all these
cases to the even symmetry odd number of taps case. The desired frequency
response in each band, as well as the weights, can be defined as arbitrary
functions. The library can use double-precision IEEE 754 floating-point numbers
for calculations, as well as other higher precision floating-point
implementations, such as
[num-bigfloat](https://docs.rs/num-bigfloat/latest/num_bigfloat/). This can be
used to solve numerically challenging problems that are difficult to solve using
double-precision arithmetic.

The implementation draws ideas from
[a paper by S.I. Filip](https://dl.acm.org/doi/10.1145/2904902)
to make the algorithm robust against numerical errors. These ideas include the
use of Chebyshev proxy root finding to find the extrema of the weighted error
function in the Remez exchange step.

## Documentation

The documentation for the Rust crate is hosted in
[docs.rs/pm-remez](https://docs.rs/pm-remez).

The documentation for the Python package is hosted in
[pm-remez.readthedocs.io](https://pm-remez.readthedocs.io/).

The Python package documentation contains a series of examples that show how to
use pm-remez to design commonly used types of FIR filters. These illustrate the
capabilities of pm-remez and also serve as a filter design guide. The
documentation of the Rust crate contains a few examples of the Rust API. The
Python examples can also be written in Rust (and in fact this is done
[as part of integration testing](tests/filter_designs.rs)).

## Python package

The pm-remez Python package is [published in
PyPI](https://pypi.org/project/pm-remez/). There are pre-built binary packages
for common architectures and operating systems. For these, the package can be
installed by doing

```
pip install pm-remez
```

## Building

The pm-remez crate uses [ndarray-linalg](https://docs.rs/ndarray-linalg/) to
solve eigenvalue problems. This in turn depends on LAPACK. The pm-remez crate
has several feature flags that are used to select the LAPACK backend. Exactly
one of these features needs to be enabled to build pm-remez. The feature flags
are `openblas-static`, `openblas-system`, `netlib-static`, `netlib-system`,
`intel-mkl-static` and `intel-mkl-system`. The `-static` versions of each flag
build the LAPACK backend and link statically against it. The `-system` versions
link against a system-installed library (linking can be dynamic or static
depending on which type of library is installed). For example,
```
cargo build --release --features openblas-system
```
will build against a system-installed OpenBLAS library.

The Python package is built using [maturin](https://docs.rs/ndarray-linalg/).
It can be built with
```
maturin build --release
```
or
```
python -mbuild
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
