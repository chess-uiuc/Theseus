#!/usr/bin/env python3
"""Reference-generation reproducibility and independent analytic polynomial checks."""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import unittest
import helmholtz_channel_difference_reference as reference


class ReferenceTests(unittest.TestCase):
    def test_checked_in_references_regenerate_exactly(self):
        directory = Path(__file__).resolve().parent
        cases = json.loads((directory/'helmholtz_channel_difference_inputs.json').read_text())['cases']
        self.assertEqual(reference.generate(cases),
                         (directory/'helmholtz_channel_difference_cases.hpp').read_text())

    def test_analytic_polynomial_all_difference_and_limit_combinations(self):
        # H=2r+r^2/2+rT/4-T^2; closed-form means independent of Hermite machinery.
        jets = []
        for property in range(4):
            for t in (2., 5.):
                for r in (.5, 2.):
                    jets.append((2*r+r*r/2+r*t/4-t*t, 2+r+t/4, r/4-2*t, .25)[property])
        for r1, t1, r2, t2 in ((.875, 2.5, 1.625, 4.25), (.875, 2.5, .875, 4.25),
                               (.875, 2.5, 1.625, 2.5), (.875, 2.5, .875, 2.5)):
            case = {'bounds': [.5, 2., 2., 5.], 'states': [r1, t1, r2, t2], 'jets': jets}
            with self.subTest(states=case['states']), localcontext() as context:
                context.prec = 80
                a, t, b, u = map(D.from_float, (r1, t1, r2, t2))
                pr = (1/t+1/u)*(2*(a+b)+a*a+a*b+b*b)/2 + (a+b)/4
                gr = (1/t+1/u)*(4+D('1.5')*(a+b))/2 + D('.5')
                pb = (2*a*a+a*a*a+2*b*b+b*b*b)/2
                gb = (4*a+D('1.5')*a*a+4*b+D('1.5')*b*b)/2 + t*u
                expected = pr/gr, gb-pb*gr/pr
                actual = reference.reference(case)
                for value, truth in zip(actual, expected):
                    self.assertLess(abs(value-truth), D('1e-65')*max(D(1), abs(truth)))


if __name__ == '__main__':
    unittest.main()
