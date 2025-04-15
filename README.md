# VCasT-tests

This repository contains a collection of test datasets and example workflows for use with [VCasT](https://github.com/NOAA-GSL/VCasT). It is maintained to illustrate and validate core functionalities of VCasT, acting as a reference for:
- Sample configuration files to demonstrate real or synthetic use cases.
- Example scripts illustrating typical tasks such as data processing, statistical analysis, and plotting.
- Test datasets that VCasT can process during automated tests or manual exploration.

### Getting Started

1. Clone This Repository

```bash
git clone https://github.com/NOAA-GSL/VCasT-tests.git
```

2. Install or Link VCasT

Ensure that VCasT is installed or accessible in your environment. This repository's examples assume you have a working VCasT setup.

3. Explore the Examples

- Look inside examples/ for small demos illustrating how to configure and run VCasT.
- Use the datasets in this repository to run tests locally or within a CI environment.

4. Run Tests

```bash
cd VCasT-tests
pytest -v
```

This command runs the entire test suite, ensuring that each dataset and example is validated against the latest VCasT code.

### License

Unless otherwise specified, the contents of this repository are covered by the same [Apache License 2.0](https://github.com/NOAA-GSL/VCasT/blob/develop/LICENSE) applied in VCasT.


