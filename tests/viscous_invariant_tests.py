#!/usr/bin/env python3
"""Exercise fail-fast checks through real subprocess termination, with NDEBUG."""
import subprocess
import sys


def main():
    executable = sys.argv[1]
    cases = {
        "valid": None,
        "bad_dimension": "layout_capacity",
        "bad_mass_index": "layout_fields",
        "bad_energy_index": "layout_fields",
        "bad_momentum_index": "momentum_index",
        "excess_equations": "layout_capacity",
        "bad_table_shape": "table_shape",
        "table_offset_overflow": "table_index_overflow",
        "bad_table_property": "table_property_indices",
        "null_table": "null_table_pointer",
        "tiny_mass": "nonzero_mass_tensor",
        "nan_mass": "nonzero_mass_tensor",
        "bad_projection": "nonzero_mass_projection",
    }
    for mode, expected in cases.items():
        result = subprocess.run(
            [executable, mode], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
        if expected is None:
            passed = result.returncode == 0 and "VISCOUS_INVARIANT_FAILURE" not in result.stdout
        else:
            passed = (
                result.returncode != 0
                and "VISCOUS_INVARIANT_FAILURE" in result.stdout
                and "reason=" + expected in result.stdout
            )
        print(f"{'PASS' if passed else 'FAIL'} {mode}")
        if not passed:
            print(result.stdout)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
