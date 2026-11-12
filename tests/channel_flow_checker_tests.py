#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import channel_flow_regression as channel


class ChannelCheckerTests(unittest.TestCase):
    def test_significant_field_difference_is_reported(self):
        reference = {'Density': np.ones(3)}
        actual = {'Density': np.array([1., 1.+1e-6, 1.])}
        report, failed = channel.compare(reference, actual)
        self.assertEqual(failed, ['Density'])
        self.assertGreater(report['Density']['max_absolute'], 1e-7)

    def test_nonfinite_and_shape_changes_are_failures(self):
        for values in [np.array([float('nan')]), np.ones(2)]:
            with self.assertRaises(AssertionError):
                channel.compare({'Density': np.ones(1)}, {'Density': values})

    def test_transport_selection_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'CMakeCache.txt'
            for value, expected in [('ON', 'sutherland'), ('OFF', 'tabulated')]:
                path.write_text(f'SUTHERLAND:BOOL={value}\n')
                self.assertEqual(channel.transport(directory), expected)
            path.write_text('')
            with self.assertRaises(ValueError):
                channel.transport(directory)

    def test_cpu_baseline_cannot_accept_cuda_run(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results/'run.log').write_text('Device configuration: cuda,cpu\n')
            with self.assertRaises(AssertionError):
                channel.frames(results, 'cpu')

    def test_record_never_overwrites_existing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            build = Path(directory)/'build'
            results = Path(directory)/'results'
            golden = source/'TestCases/GoldenData/ChannelFlow'
            for path in (build, results, golden):
                path.mkdir(parents=True)
            (build/'CMakeCache.txt').write_text('SUTHERLAND:BOOL=OFF\n')
            (results/'run.log').write_text('Theseus Simulation with 1 MPI ranks\n')
            reference = golden/'tabulated.npz'
            reference.write_bytes(b'existing approved data')
            with patch.object(channel, 'frames', return_value=([], np.zeros((50, 3)), 1.0)):
                with self.assertRaises(FileExistsError):
                    channel.record(source, build, results, 'test')
            self.assertEqual(reference.read_bytes(), b'existing approved data')

    def test_case_mesh_has_two_periodic_directions_and_wall_names(self):
        case = Path(__file__).resolve().parents[1]/'TestCases/LTE/NavierStokes/ChannelFlowRegression'
        spec = importlib.util.spec_from_file_location('channel_mesh', case/'make_mesh.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory)/'channel.msh'
            generator.write_mesh(mesh)
            text = mesh.read_text()
            self.assertEqual(text, (case/'channel_4x4x4.msh').read_text())
            self.assertIn('2 3 "BottomWall"', text)
            self.assertIn('2 4 "TopWall"', text)
            periodic = text.split('$Periodic\n', 1)[1].splitlines()
            self.assertEqual(periodic[0], '2')
            self.assertIn('2 1 2', periodic)
            self.assertIn('2 5 6', periodic)
            self.assertNotIn('2 3 4', periodic)
            elements = text.split('$Elements\n', 1)[1].split('$EndElements')[0].splitlines()[1:]
            self.assertEqual(sum(line.split()[1] == '5' for line in elements), 64)


if __name__ == '__main__':
    unittest.main()
