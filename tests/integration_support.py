"""File/log handling for integrated checks; run_theseus owns platform launch policy."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile


@contextmanager
def work_directory(prefix):
    directory = tempfile.mkdtemp(prefix=prefix, dir=Path.cwd())
    print(f"Integration files: {directory}", flush=True)
    yield directory


def run_simulation(executable, config, ranks, device, timeout=120):
    source = Path(__file__).resolve().parents[1]
    config = Path(config).resolve()
    launch = config.parent / (config.stem + "-launch")
    command = ["bash", str(source / "scripts/run_theseus.sh"),
               "-e", str(Path(executable).resolve()), "-c", str(config),
               "-o", str(launch), "-p", str(ranks), "-r", device, "-P", "-R", "-k"]
    log_path = config.with_suffix(".run.log")
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=source, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    print(f"Simulation log: {log_path}", flush=True)
    return subprocess.CompletedProcess(command, result.returncode,
                                       stdout=log_path.read_text(), stderr="")
