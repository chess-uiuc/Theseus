#!/usr/bin/env python3
"""Focused regression tests for scripts/run_theseus.sh."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "run_theseus.sh"
FRONTERA_LAUNCHER = ROOT / "scripts" / "launch_frontera_device.sh"


class HarnessFixture:
    def __init__(self, tempdir: str):
        self.root = Path(tempdir)
        self.bindir = self.root / "bin"
        self.builddir = self.root / "build"
        self.scriptsdir = self.root / "scripts"
        self.bindir.mkdir()
        self.builddir.mkdir()
        self.scriptsdir.mkdir()
        shutil.copy(FRONTERA_LAUNCHER, self.scriptsdir)

        self.config = self.root / "Case" / "config.json"
        self.config.parent.mkdir()
        self.config.write_text(
            json.dumps(
                {
                    "runTime": {
                        "visualize": True,
                        "paraview": True,
                        "visit": False,
                        "output_file_path": "unused",
                        "checkpoint_load": False,
                        "mesh_file": "mesh.file",
                    }
                }
            ),
            encoding="utf-8",
        )

        self._write_launcher("mpiexec", 2)
        self._write_launcher("custom-launcher", 2)
        self._write_launcher("ibrun", 2)
        self._write_launcher("flux", 6)
        self._write_hostname()
        self._write_theseus()

    def _write_launcher(self, name: str, shift_count: int) -> None:
        launcher = self.bindir / name
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '{name}' >> \"$LAUNCH_LOG\"\n"
            "printf ' <%s>' \"$@\" >> \"$LAUNCH_LOG\"\n"
            "printf '\\n' >> \"$LAUNCH_LOG\"\n"
            f"shift {shift_count}\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)

    def _write_hostname(self) -> None:
        hostname = self.bindir / "hostname"
        hostname.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$TEST_HOSTNAME\"\n",
            encoding="utf-8",
        )
        hostname.chmod(0o755)

    def _write_theseus(self) -> None:
        executable = self.builddir / "theseus"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "config_path = pathlib.Path(sys.argv[sys.argv.index('-c') + 1])\n"
            "config = json.loads(config_path.read_text())\n"
            "pathlib.Path(os.environ['PATCHED_CONFIG_COPY']).write_text(json.dumps(config))\n"
            "mode = os.environ.get('FAKE_OUTPUT_MODE', 'none')\n"
            "pv = pathlib.Path('ParaView')\n"
            "if mode != 'none':\n"
            "    pv.mkdir()\n"
            "    cycle0 = pv / 'Cycle000000'\n"
            "    cycle0.mkdir()\n"
            "    (cycle0 / 'data.pvtu').touch()\n"
            "    datasets = ['Cycle000000/data.pvtu']\n"
            "    if mode == 'complete':\n"
            "        final_step = config['runTime'].get('nsteps_max', 7)\n"
            "        final = pv / f'Cycle{final_step:06d}'\n"
            "        final.mkdir(exist_ok=True)\n"
            "        (final / 'data.pvtu').touch()\n"
            "        datasets.append(f'Cycle{final_step:06d}/data.pvtu')\n"
            "    body = ''.join(f'<DataSet file=\\\"{item}\\\"/>' for item in datasets)\n"
            "    (pv / 'ParaView.pvd').write_text(f'<VTKFile><Collection>{body}</Collection></VTKFile>')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    def run(self, *args: str, hostname: str = "workstation", output: str = "none",
            launcher: str | None = None):
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{self.bindir}:{env['PATH']}",
                "PYTHON": sys.executable,
                "TEST_HOSTNAME": hostname,
                "FAKE_OUTPUT_MODE": output,
                "LAUNCH_LOG": str(self.root / "launcher.log"),
                "PATCHED_CONFIG_COPY": str(self.root / "patched-config.json"),
                "NGPU": "4",
            }
        )
        env.pop("THESEUS_MPI_LAUNCHER", None)
        if launcher is not None:
            env["THESEUS_MPI_LAUNCHER"] = launcher
        return subprocess.run(
            ["/bin/bash", str(HARNESS), "-b", str(self.builddir), *args, "-c", str(self.config)],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def checkpoint(self, cycle, *, ranks=2, folder="Checkpoints", metadata=True,
                   files=None, empty_rank=None, extra_metadata=None):
        directory = self.root / "RunTheseus/Case" / folder / f"Cycle{cycle}"
        directory.mkdir(parents=True, exist_ok=True)
        if metadata:
            values = {"cycle": cycle, "time": cycle * 0.001, "mpi_ranks": ranks}
            values.update(extra_metadata or {})
            (directory / f"checkpoint_cycle_{cycle}.json").write_text(json.dumps(values))
        for rank in range(ranks if files is None else files):
            data = b"" if rank == empty_rank else b"checkpoint fixture"
            (directory / f"checkpoint_cycle_{cycle}.{rank:08d}.chk").write_bytes(data)
        return directory

    def patched_config(self):
        return json.loads((self.root / "patched-config.json").read_text())

    def launcher_log(self):
        return (self.root / "launcher.log").read_text()


class RunTheseusHarnessTests(unittest.TestCase):
    def test_k_disables_check_without_disabling_visualization(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            result = fixture.run("-k", output="none")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(fixture.patched_config()["runTime"]["visualize"])

    def test_z_disables_visualization_and_output_check(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            result = fixture.run("-z", output="none")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertFalse(fixture.patched_config()["runTime"]["visualize"])
            self.assertFalse((fixture.root / "RunTheseus" / "Case" / "ParaView").exists())

    def test_restart_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            config = json.loads(fixture.config.read_text())
            config["runTime"]["checkpoint_load"] = True
            fixture.config.write_text(json.dumps(config), encoding="utf-8")

            result = fixture.run("-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertFalse(fixture.patched_config()["runTime"]["checkpoint_load"])

    def test_R_preserves_restart_configuration(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            config = json.loads(fixture.config.read_text())
            config["runTime"]["checkpoint_load"] = True
            fixture.config.write_text(json.dumps(config), encoding="utf-8")

            result = fixture.run("-R", "-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(fixture.patched_config()["runTime"]["checkpoint_load"])

    def test_latest_selects_numeric_highest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HarnessFixture(temporary)
            fixture.checkpoint(9)
            fixture.checkpoint(10)
            fixture.checkpoint(100, files=1)
            fixture.checkpoint(101, metadata=False)
            fixture.checkpoint(102, empty_rank=1)
            malformed = fixture.checkpoint(103)
            (malformed / "checkpoint_cycle_103.json").write_text("{broken")
            result = fixture.run("-L", "-k")
            self.assertEqual(result.returncode, 0, result.stdout)
            runtime = fixture.patched_config()["runTime"]
            self.assertTrue(runtime["checkpoint_load"])
            self.assertEqual(runtime["checkpoint_cycle"], 10)
            self.assertTrue((fixture.root / "RunTheseus/Case/Checkpoints/Cycle100").exists())

    def test_explicit_cycle_overrides_json_and_does_not_select_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HarnessFixture(temporary)
            fixture.checkpoint(3)
            fixture.checkpoint(20)
            config = json.loads(fixture.config.read_text())
            config["runTime"].update(checkpoint_load=False, checkpoint_cycle=40)
            fixture.config.write_text(json.dumps(config))
            result = fixture.run("-C", "3", "-k")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(fixture.patched_config()["runTime"]["checkpoint_cycle"], 3)
            self.assertTrue(fixture.patched_config()["runTime"]["checkpoint_load"])

    def test_latest_honors_custom_output_and_checkpoint_folder_with_P(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HarnessFixture(temporary)
            fixture.checkpoint(12, folder="unused/custom-checkpoints")
            config = json.loads(fixture.config.read_text())
            config["runTime"]["checkpoints_folder"] = "custom-checkpoints"
            fixture.config.write_text(json.dumps(config))
            result = fixture.run("-P", "-L", "-k")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(fixture.patched_config()["runTime"]["checkpoint_cycle"], 12)

    def test_latest_skips_incompatible_rank_count_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HarnessFixture(temporary)
            fixture.checkpoint(10, extra_metadata={"order": 2})
            fixture.checkpoint(20, extra_metadata={"order": 3})
            fixture.checkpoint(30, ranks=3)
            result = fixture.run("-L", "-y", "2", "-k")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(fixture.patched_config()["runTime"]["checkpoint_cycle"], 10)

    def test_selection_failure_never_launches_solver(self):
        for selection in (("-L",), ("-C", "5")):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                fixture = HarnessFixture(temporary)
                fixture.checkpoint(5, files=1)
                result = fixture.run(*selection, "-k")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((fixture.root / "launcher.log").exists())

    def test_explicit_missing_cycle_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = HarnessFixture(temporary)
            fixture.checkpoint(10)
            result = fixture.run("-C", "20", "-k")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse((fixture.root / "launcher.log").exists())

    def test_selectors_reject_wipe_before_removing_checkpoint_files(self):
        for selection in (("-L",), ("-C", "10")):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                fixture = HarnessFixture(temporary)
                directory = fixture.checkpoint(10)
                result = fixture.run(*selection, "-w", "-k")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertTrue(directory.exists())
                self.assertFalse((fixture.root / "launcher.log").exists())

    def test_invalid_or_conflicting_selectors_are_rejected(self):
        for selection in (("-C", "0"), ("-C", "-1"), ("-C", "abc"),
                          ("-L", "-C", "10"), ("-C", "10", "-L")):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                fixture = HarnessFixture(temporary)
                result = fixture.run(*selection, "-k")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse((fixture.root / "launcher.log").exists())

    def test_mesh_filename_keeps_configured_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            config = json.loads(fixture.config.read_text())
            config["runTime"]["mesh_file"] = "configured/mesh.file"
            fixture.config.write_text(json.dumps(config), encoding="utf-8")

            result = fixture.run("-m", "replacement.msh", "-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(
                fixture.patched_config()["runTime"]["mesh_file"],
                "configured/replacement.msh",
            )

    def test_relative_mesh_path_is_resolved_from_launch_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)

            result = fixture.run("-m", "user-meshes/replacement.msh", "-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(
                fixture.patched_config()["runTime"]["mesh_file"],
                str(fixture.root.resolve() / "user-meshes" / "replacement.msh"),
            )

    def test_absolute_mesh_path_is_used_unchanged(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            mesh = fixture.root / "external-meshes" / "replacement.msh"

            result = fixture.run("-m", str(mesh), "-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(
                fixture.patched_config()["runTime"]["mesh_file"], str(mesh)
            )

    def test_cfl_override_enables_variable_timestep(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            config = json.loads(fixture.config.read_text())
            config["runTime"]["variable_dt"] = False
            config["runTime"]["dt"] = 0.25
            fixture.config.write_text(json.dumps(config), encoding="utf-8")

            result = fixture.run("-s", "0.125", "-k")

            self.assertEqual(result.returncode, 0, result.stdout)
            runtime = fixture.patched_config()["runTime"]
            self.assertTrue(runtime["variable_dt"])
            self.assertEqual(runtime["cfl"], 0.125)

    def test_fixed_timestep_and_cfl_overrides_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)

            result = fixture.run("-t", "0.01", "-s", "0.1", "-k")

            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("choose either -t TIMESTEP or -s CFL", result.stdout)

    def test_explicit_step_check_requires_final_cycle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            result = fixture.run("-n", "7", output="initial")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("Cycle000007", result.stdout)

    def test_final_time_check_requires_distinct_existing_datasets(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            result = fixture.run(output="initial")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("distinct first and last datasets", result.stdout)

            result = fixture.run(output="complete")
            self.assertEqual(result.returncode, 0, result.stdout)

    def test_platform_launch_commands_are_preserved(self):
        cases = [
            ("workstation", ("mpiexec", "<-n> <5>")),
            ("tuo123", ("flux", "<run> <--exclusive> <-N> <3> <-n> <5>")),
            ("front123", ("ibrun", "<-n> <5> <../device_launcher.sh>")),
            ("c1-2", ("ibrun", "<-n> <5> <../device_launcher.sh>")),
        ]
        for hostname, expected in cases:
            with self.subTest(hostname=hostname), tempfile.TemporaryDirectory() as tempdir:
                fixture = HarnessFixture(tempdir)
                extra = ("-k", "-p", "5")
                if hostname.startswith("tuo"):
                    extra += ("-H", "3")
                result = fixture.run(*extra, hostname=hostname)
                self.assertEqual(result.returncode, 0, result.stdout)
                log = fixture.launcher_log()
                self.assertTrue(log.startswith(expected[0]), log)
                self.assertIn(expected[1], log)

    def test_one_explicit_launcher_override_precedes_platform_detection(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fixture = HarnessFixture(tempdir)
            launcher = str(fixture.bindir / "custom-launcher")
            result = fixture.run("-k", "-p", "5", hostname="front123",
                                 launcher=launcher)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(fixture.launcher_log().startswith("custom-launcher"))


if __name__ == "__main__":
    unittest.main()
