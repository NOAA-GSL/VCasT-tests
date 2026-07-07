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

def compare_png_images(path1, path2):
    """Compare two PNG images using pixel-wise difference."""
    try:
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if img1.size != img2.size:
            return False, f"Image dimensions differ: {img1.size} vs {img2.size}"

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        equal = np.array_equal(arr1, arr2)
        if not equal:
            diff = np.abs(arr1 - arr2)
            mismatch = np.count_nonzero(diff)
            return False, f"Images differ in {mismatch} pixels"
        return True, ""
    except Exception as e:
        return False, f"Error comparing PNGs: {e}"

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
                equal, msg = compare_png_images(expected_path, output_path)
                assert equal, f"PNG mismatch: {msg}"
            else:
                # Assume text or binary
                if not os.path.exists(expected_path):
                    raise FileNotFoundError(f"Expected file not found: {expected_path}")

                if not filecmp.cmp(expected_path, output_path, shallow=False):
                    diff = get_file_diff(expected_path, output_path)
                    assert False, f"Text/binary file mismatch:\n{diff}"

_test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.yaml")

@pytest.mark.parametrize("test_case", yaml.safe_load(open(_test_cases_path))["tests"])
def test_dynamic_cases(test_case):
    run_test_case(test_case)

