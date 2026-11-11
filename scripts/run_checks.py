"""Run the full test suite without pytest:

    python scripts/run_checks.py

Every test_* function in tests/ is executed; the exit code is non-zero if
any test fails.  With pytest installed, `pytest tests/` works as well.
"""

import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    t0 = time.time()
    modules = ["test_topology", "test_statics", "test_modal",
               "test_scenarios", "test_records"]
    passed, failed = 0, 0
    for mod_name in modules:
        print(f"\n--- {mod_name} " + "-" * (50 - len(mod_name)))
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            traceback.print_exc()
            failed += 1
            continue
        tests = sorted(k for k in vars(mod) if k.startswith("test_"))
        for name in tests:
            fn = getattr(mod, name)
            try:
                fn()
                print(f"  [PASS] {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  [FAIL] {name}: {exc}")
                failed += 1
            except Exception:
                print(f"  [ERROR] {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'=' * 64}")
    print(f"{passed} passed, {failed} failed in {time.time() - t0:.1f} s")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
