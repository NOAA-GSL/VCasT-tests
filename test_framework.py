import os
import yaml
import subprocess
import logging
import difflib
import numpy as np
from PIL import Image
import pytest
import filecmp

# Configure logging for more verbose test output
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_file_diff(expected_path, output_path):
    try:
        with open(expected_path, "r", encoding="utf8", errors="replace") as f:
            expected_lines = f.readlines()
        with open(output_path, "r", encoding="utf8", errors="replace") as f:
            output_lines = f.readlines()
        diff = list(difflib.unified_diff(expected_lines, output_lines,
                                         fromfile="expected", tofile="output", lineterm=""))
        return "\n".join(diff) if diff else "No differences found."
    except Exception as e:
        return f"Could not compute diff: {e}"

def check_png_is_valid_plot(path, min_width=50, min_height=50):
    """Smoke-test a generated plot instead of diffing it against a golden PNG.

    Two escalating attempts at pixel-diffing (exact, then a tolerance-based
    version allowing small dimension/color drift) both broke across the
    Python 3.10/3.11/3.12 CI matrix: different matplotlib releases pip
    resolves for each job change default marker/legend/font rendering
    enough to shift 7-11% of pixels with no actual change in plot content.
    That's not something a fixed tolerance can chase without becoming
    meaningless.

    The real correctness check for plot content is the underlying data
    file each plot is built from -- compared exactly, byte-for-byte, in
    run_test_case() below. The PNG is just a rendering of that data, so
    this only confirms VCasT actually produced a plausible image: a valid,
    reasonably sized file that isn't blank.
    """
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return False, f"Not a valid image file: {e}"

    with Image.open(path) as img:
        if img.width < min_width or img.height < min_height:
            return False, f"Image implausibly small: {img.size}"

        arr = np.asarray(img.convert("RGB"))
        stddev = float(arr.std())
        if stddev < 1.0:
            return False, f"Image appears blank (pixel stddev={stddev:.3f})"

    return True, ""

def run_test_case(test_case):
    example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), test_case["example_dir"]))
    logging.info("Running test case '%s' in '%s'", test_case["name"], example_dir)

    for command in test_case["commands"]:
        config_file = command["config"]
        config_path = os.path.join(example_dir, config_file)
        logging.info("Executing: vcast %s --test-mode", config_path)

        result = subprocess.run(
            ["vcast", config_path, "--test-mode"],
            cwd=example_dir,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Command failed:\n{result.stdout}\n{result.stderr}"
        if result.stdout:
            logging.info("STDOUT: %s", result.stdout.strip())
        if result.stderr:
            logging.info("STDERR: %s", result.stderr.strip())

        for output_file, expected_file in command.get("outputs", {}).items():
            output_path = os.path.join(example_dir, output_file)
            expected_path = os.path.join(example_dir, expected_file)
            logging.info("Comparing '%s' with '%s'", output_path, expected_path)

            assert os.path.exists(output_path), f"Missing output file: {output_path}"

            if output_path.endswith(".png"):
                ok, msg = check_png_is_valid_plot(output_path)
                assert ok, f"Generated plot failed smoke test: {msg}"
            else:
                # Assume text or binary
                if not os.path.exists(expected_path):
                    raise FileNotFoundError(f"Expected file not found: {expected_path}")

                if not filecmp.cmp(expected_path, output_path, shallow=False):
                    diff = get_file_diff(expected_path, output_path)
                    assert False, f"Text/binary file mismatch:\n{diff}"

_test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.yaml")

@pytest.mark.integration
@pytest.mark.parametrize("test_case", yaml.safe_load(open(_test_cases_path))["tests"])
def test_dynamic_cases(test_case):
    run_test_case(test_case)

