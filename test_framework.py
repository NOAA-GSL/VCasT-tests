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

def compare_png_images(path1, path2, pixel_tolerance=12, max_mismatch_fraction=0.01,
                        max_size_drift=15):
    """Compare two PNG images allowing for minor rendering differences.

    matplotlib plots are re-rendered from scratch by each environment that
    runs the tests, and small differences in matplotlib/freetype/font
    versions shift anti-aliasing and (via bbox_inches='tight') even the
    final canvas size by a few pixels, with no actual change in plot
    content. Exact pixel/size equality is too strict to survive that kind
    of environment drift, so this allows:
      - up to `max_size_drift` pixels of difference in either dimension
        (larger differences likely mean real content changed, so those
        still fail outright, compared over the shared top-left region);
      - per-channel color differences up to `pixel_tolerance` (absorbs
        anti-aliasing noise);
      - up to `max_mismatch_fraction` of pixels still differing beyond
        that tolerance (catches remaining minor rendering noise while
        still failing on genuine content differences, which touch a much
        larger fraction of the image).
    """
    try:
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if img1.size != img2.size:
            size_diff = max(abs(img1.width - img2.width), abs(img1.height - img2.height))
            if size_diff > max_size_drift:
                return False, f"Image dimensions differ: {img1.size} vs {img2.size}"
            w, h = min(img1.width, img2.width), min(img1.height, img2.height)
            img1 = img1.crop((0, 0, w, h))
            img2 = img2.crop((0, 0, w, h))

        arr1 = np.asarray(img1, dtype=np.int16)
        arr2 = np.asarray(img2, dtype=np.int16)

        diff = np.abs(arr1 - arr2)
        mismatched_pixels = int(np.count_nonzero(np.any(diff > pixel_tolerance, axis=-1)))
        total_pixels = arr1.shape[0] * arr1.shape[1]
        mismatch_fraction = mismatched_pixels / total_pixels

        if mismatch_fraction > max_mismatch_fraction:
            return False, (f"Images differ in {mismatched_pixels}/{total_pixels} pixels "
                            f"({mismatch_fraction:.2%}, tolerance {max_mismatch_fraction:.2%})")
        return True, ""
    except Exception as e:
        return False, f"Error comparing PNGs: {e}"

def run_test_case(test_case):
    example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", test_case["example_dir"]))
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

@pytest.mark.parametrize("test_case", yaml.safe_load(open("tests/test_cases.yaml"))["tests"])
def test_dynamic_cases(test_case):
    run_test_case(test_case)

