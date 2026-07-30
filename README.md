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

2. Install VCasT (with the `dev` extra, which pulls in `pytest` and `Pillow`)

```bash
pip install -e "/path/to/VCasT[all,dev]"
```

The `vcast` console script must be on `PATH` and importable before running any tests here.

3. Explore the Examples

- Look inside examples/ for small demos illustrating how to configure and run VCasT.
- Use the datasets in this repository to run tests locally or within a CI environment.

4. Run Tests

```bash
cd VCasT-tests
pytest -v
```

This runs two layers of tests:
- `test_stats/` and `test_classes/`: unit tests that call `vcast` functions/classes directly.
- `test_framework.py`: integration tests driven by `test_cases.yaml` that invoke the `vcast` CLI end-to-end and compare outputs against the fixtures under `examples/*/expected_outputs/`.

This works whether the repository is checked out standalone or embedded as the `tests/` submodule of VCasT.

### License

Unless otherwise specified, the contents of this repository are covered by the same [Apache License 2.0](https://github.com/NOAA-GSL/VCasT/blob/develop/LICENSE) applied in VCasT.


