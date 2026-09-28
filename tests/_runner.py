"""Tiny runner so every test file also works as `python tests/test_x.py`."""
import sys


def run(namespace):
    tests = [v for k, v in sorted(namespace.items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
