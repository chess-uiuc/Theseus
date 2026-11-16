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
    'mms-verification.md', 'mms-evaluations.md', 'mms-hpc.md', 'mms-inventory.md'))


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
        self.assertEqual(expand(selections[2]), list(range(90)))
        for task in pilots:
            self.assertEqual(((task % 30)//6 + 1, task % 6), (3, 1))

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
