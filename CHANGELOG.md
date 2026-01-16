# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4] - 2026-01-16

### Fixed

- Fixed typos in Python package documentation.

## [0.2.3] - 2026-01-15

### Changed

- Updated dependencies.
- Minor code improvements.
- Changed default features from `faer-backend` to `nalgebra-backend` to match
  the eigenvalue backend used in the Python package.

## [0.2.2] - 2025-12-02

### Changed

- Updated dependencies.

## [0.2.1] - 2025-09-02

### Added

- Added documentation about calling response and weight closures outside bands.

### Changed

- Updated dependencies.

## [0.2.0] - 2025-07-31

### Added

- Support for multiple eigenvalue backends. faer and nalgebra backends added.

### Changed

- Use faer as default eigenvalue for Rust crate.
- Use nalgebra as default eigenvalue backend for Python package.
- Updated sphinx documentation dependencies.
- Updated dependencies.

## [0.1.8] - 2025-04-13

### Changed

- Updated dependencies to fix rust-openssl and pyo3 security advisories.

## [0.1.7] - 2025-03-25

### Changed

- Updated dependencies to fix ring security advisory.
- Update to Rust edition 2024.

## [0.1.6] - 2025-02-10

### Changed

- Updated dependencies to fix openssl security advisory.

## [0.1.5] - 2025-01-30

### Changed

- Updated ndarray-linalg to v0.17.0, removing the need to patch lax to build on arm
- Updated other dependencies.

## [0.1.4] - 2024-12-05

### Changed

- Updated dependecies. This fixes some security warnings.

## [0.1.3] - 2024-10-09

### Fixed

- Patch lax crate to allow building on linux aarch64.

## [0.1.2] - 2024-10-08

### Added

- Ichige FIR order estimate.

### Changed

- Updated dependencies.

## [0.1.1] - 2024-09-07

### Changed

- Updated dependencies.

## [0.1.0] - 2024-04-09

### Added

- First release of pm-remez.

[unreleased]: https://github.com/maia-sdr/pm-remez/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/maia-sdr/pm-remez/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/maia-sdr/pm-remez/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/maia-sdr/pm-remez/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/maia-sdr/pm-remez/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/maia-sdr/pm-remez/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/maia-sdr/pm-remez/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/maia-sdr/pm-remez/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/maia-sdr/pm-remez/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/maia-sdr/pm-remez/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/maia-sdr/pm-remez/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/maia-sdr/pm-remez/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/maia-sdr/pm-remez/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/maia-sdr/pm-remez/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maia-sdr/pm-remez/releases/tag/v0.1.0
