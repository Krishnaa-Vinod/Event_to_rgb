#!/usr/bin/env python3
"""Basic functionality test for the fixed Event_to_rgb pipeline."""

import subprocess
import sys
import os
from pathlib import Path

def test_inspect_inputs():
    """Test the fixed inspect_inputs.py script."""
    print("Testing inspect_inputs.py...")

    try:
        result = subprocess.run([
            "python", "scripts/inspect_inputs.py",
            "--bag-dir", "/scratch/kvinod/bags/overfitting_data/data_collect_20260228_153433",
            "--h5-file", "/scratch/kvinod/bags/eGo_navi_overfit_data_h5/data_collect_20260228_153433.h5"
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✓ inspect_inputs.py works correctly")
            print("Output snippet:", result.stdout[:200] + "...")
            return True
        else:
            print("✗ inspect_inputs.py failed")
            print("Error:", result.stderr[:200])
            return False
    except Exception as e:
        print(f"✗ inspect_inputs.py error: {e}")
        return False

def test_paths_config():
    """Test paths config loading."""
    print("\nTesting paths config loading...")

    # Test run_all.py help (should not crash)
    try:
        result = subprocess.run([
            "python", "scripts/run_all.py", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and "paths-config" in result.stdout:
            print("✓ run_all.py CLI and paths config support working")
            return True
        else:
            print("✗ run_all.py CLI test failed")
            return False
    except Exception as e:
        print(f"✗ run_all.py CLI error: {e}")
        return False

def test_config_file():
    """Test that config file exists and is valid."""
    print("\nTesting config file...")

    config_path = Path("configs/paths.json")
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)

            if "bag_dir" in config and "h5_file" in config:
                print("✓ paths.json config file is valid")
                print(f"  bag_dir: {config['bag_dir']}")
                print(f"  h5_file: {config['h5_file']}")
                return True
            else:
                print("✗ paths.json missing required keys")
                return False
        except Exception as e:
            print(f"✗ paths.json parsing error: {e}")
            return False
    else:
        print("✗ configs/paths.json not found")
        return False

def main():
    print("=== Event-to-RGB Basic Functionality Test ===")

    tests = [
        test_inspect_inputs,
        test_paths_config,
        test_config_file
    ]

    results = []
    for test in tests:
        results.append(test())

    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("Key fixes verified:")
        print("  - inspect_inputs.py now takes CLI arguments")
        print("  - run_all.py supports paths config")
        print("  - Core argument parsing works correctly")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)