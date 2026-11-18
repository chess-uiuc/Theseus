"""Keep MMS procedures executable and public numerical evidence portable."""
from mms_paths import REPOSITORY_ROOT
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = REPOSITORY_ROOT
DOCUMENTS = tuple(ROOT/'docs'/name for name in (
    'mms-verification.md', 'mms-evaluations.md', 'mms-hpc.md', 'mms-inventory.md', 'mms-mesh-sensitivity.md'))


class MMSDocumentationTests(unittest.TestCase):
    def test_runtime_and_test_layout_are_separate(self):
        runtime = ROOT/'include/verification/mms'
        self.assertTrue((runtime/'TrigonometricEulerMMS.hpp').is_file())
        self.assertTrue((runtime/'sources/TrigonometricCNS.hpp').is_file())
        self.assertFalse((ROOT/'include/TrigonometricEulerMMS.hpp').exists())
        self.assertFalse((ROOT/'tests/cns_matrix_study.py').exists())
        for header in runtime.rglob('*.hpp'):
            self.assertNotRegex(header.read_text(), r'#include[^\n]*tests/')
        definitions = (ROOT/'tests/CMakeLists.txt').read_text()
        self.assertIn('mms/trigonometric_euler_tests.cpp', definitions)
        self.assertIn('mms/trigonometric_sutherland_tests.cpp', definitions)
        self.assertIn('mms/euler_rhs_probe.cpp', definitions)

    def test_local_document_links_exist(self):
        for document in DOCUMENTS:
            targets = re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', document.read_text())
            for target in targets:
                if re.match(r'^[a-z]+://', target) or target.startswith('#'):
                    continue
                path = target.split('#', 1)[0]
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent/path).is_file())

    def test_bash_examples_have_valid_syntax(self):
        for document in DOCUMENTS:
            blocks = re.findall(r'```bash\n(.*?)```', document.read_text(), re.S)
            for index, block in enumerate(blocks):
                with self.subTest(document=document.name, block=index):
                    result = subprocess.run(['bash', '-n'], input=block, text=True,
                                            capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_hpc_pilots_and_remaining_jobs_cover_matrix_once(self):
        text = (ROOT/'docs/mms-hpc.md').read_text()
        selections = re.findall(r'--array=([0-9,%-]+)', text)
        def expand(selection):
            result = []
            for span in selection.split('%', 1)[0].split(','):
                bounds = [int(value) for value in span.split('-')]
                result.extend(range(bounds[0], bounds[-1] + 1))
            return result
        pilots = expand(selections[0])
        remaining = expand(selections[1])
        self.assertEqual(pilots, [13, 43, 73])
        self.assertFalse(set(pilots) & set(remaining))
        self.assertEqual(sorted(pilots + remaining), list(range(90)))
        supersonic_main = expand(selections[2])
        supersonic_p5 = expand(selections[3])
        self.assertFalse(set(supersonic_main) & set(supersonic_p5))
        self.assertFalse(set(pilots) & (set(supersonic_main) | set(supersonic_p5)))
        self.assertEqual(sorted(pilots + supersonic_main + supersonic_p5), list(range(90)))
        self.assertTrue(all((task % 30)//6 == 4 for task in supersonic_p5))
        self.assertTrue(all((task % 30)//6 < 4 for task in supersonic_main))
        self.assertEqual(expand(selections[4]), list(range(90)))
        for task in pilots:
            self.assertEqual(((task % 30)//6 + 1, task % 6), (3, 1))

    def test_hpc_logs_stay_inside_results(self):
        text = (ROOT/'docs/mms-hpc.md').read_text()
        self.assertNotIn('mkdir -p logs', text)
        self.assertNotRegex(text, r'--output=logs/')
        self.assertIn('--output="$RESULTS_ROOT/logs/build-%j.log"', text)
        self.assertIn('--output="$RESULTS_ROOT/$regime/logs/%x-%A_%a.log"', text)
        self.assertIn('--timeout "${MMS_TIMEOUT:-1700}"', text)

    def test_three_regime_evidence_preserves_separate_control_results(self):
        assets = ROOT/'docs/assets/mms/three-regime-matrix'
        summary = json.loads((assets/'summary.json').read_text())
        groups = json.loads((assets/'groups.json').read_text())
        provenance = json.loads((assets/'provenance.json').read_text())
        diagnostic = json.loads((assets/'supplemental-timestep-diagnostic.json').read_text())
        self.assertEqual(len(groups), 270)
        self.assertEqual(len({(g['regime'], g['physics'], g['degree'], g['flux']) for g in groups}), 270)
        self.assertEqual(summary['spatial_simulations'], 1620)
        self.assertEqual(summary['timestep_control_simulations'], 270)
        self.assertEqual(summary['total_simulations'], 1890)
        self.assertEqual(sum(g['spatial_order_floor_met'] for g in groups), 226)
        self.assertEqual(sum(g['dt_control_passed'] for g in groups), 264)
        for regime, expected_orders, expected_controls in (
                ('subsonic', 48, 90), ('transonic', 88, 90), ('supersonic', 90, 84)):
            subset = [g for g in groups if g['regime'] == regime]
            self.assertEqual(len(subset), 90)
            self.assertEqual(sum(g['spatial_order_floor_met'] for g in subset), expected_orders)
            self.assertEqual(sum(g['dt_control_passed'] for g in subset), expected_controls)
            self.assertTrue(all(len(g['errors']) == 6 and len(g['observed_orders']) == 5 for g in subset))
        original_roe = next(g for g in groups if (g['regime'], g['physics'], g['degree'], g['flux'])
                            == ('supersonic', 'euler', 5, 'Roe'))
        self.assertFalse(original_roe['dt_control_passed'])
        self.assertTrue(diagnostic['assessment']['controls_passed'])
        csv_lines = (assets/'observed-orders.csv').read_text().splitlines()
        self.assertEqual(len(csv_lines), 1351)
        for filename, digest in provenance['files'].items():
            with self.subTest(asset=filename):
                self.assertEqual(hashlib.sha256((assets/filename).read_bytes()).hexdigest(), digest)

    def test_mesh_sensitivity_evidence_retains_order_flags(self):
        assets = ROOT/'docs/assets/mms/mesh-sensitivity'
        data = json.loads((assets/'measurements.json').read_text())
        provenance = json.loads((assets/'provenance.json').read_text())
        self.assertEqual(len(data['records']), 108)
        self.assertTrue(data['core_checks_passed'])
        self.assertFalse(data['passed'])
        self.assertEqual(len(data['order_flags']), 27)
        self.assertEqual({r['mesh'] for r in data['records']},
                         {'uniform', 'graded', 'rotated', 'sheared', 'jitter17', 'jitter91'})
        self.assertEqual(len(data['groups']), 36)
        for filename, digest in provenance['files'].items():
            self.assertEqual(hashlib.sha256((assets/filename).read_bytes()).hexdigest(), digest)

    def test_public_metadata_has_no_machine_absolute_paths(self):
        for path in (ROOT/'docs/assets/mms').rglob('*.json'):
            data = json.loads(path.read_text())
            def inspect(value):
                if isinstance(value, dict):
                    for item in value.values():
                        inspect(item)
                elif isinstance(value, list):
                    for item in value:
                        inspect(item)
                elif isinstance(value, str):
                    self.assertIsNone(re.match(r'^(?:/|[A-Za-z]:[\\/])', value), str(path))
            inspect(data)
        for document in DOCUMENTS:
            self.assertIsNone(re.search(r'/(?:Users|home)/[^/\s]+/|/work/nvme/',
                                        document.read_text()), document.name)

    def test_hpc_evidence_matches_inventory_and_hashes(self):
        assets = ROOT/'docs/assets/mms/hpc-matrix'
        summary = json.loads((assets/'summary.json').read_text())
        groups = json.loads((assets/'groups.json').read_text())
        provenance = json.loads((assets/'provenance.json').read_text())
        self.assertEqual(len(groups), 90)
        self.assertEqual(summary['completed_solver_logs'], 630)
        self.assertEqual(summary['assessments_present'], 90)
        self.assertEqual(len(summary['order_flags']), 2)
        self.assertTrue(all(g['component_control_verified'] for g in groups))
        self.assertEqual(len({(g['physics'], g['degree'], g['flux']) for g in groups}), 90)
        for filename, digest in provenance['files'].items():
            with self.subTest(asset=filename):
                self.assertEqual(hashlib.sha256((assets/filename).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
