"""M10 acceptance tests: CLI, thin Python wrapper, reproducibility,
destructive configs, selftest.

Criteria (docs/validation.md):
  M10-1 every examples/ config runs end-to-end via the CLI with complete
        outputs (separate full-tier test);
  M10-2 lithofem.run() output byte-identical to the CLI; dataclass-built
        YAML passes the schema;
  M10-3 reproducibility: two runs of the same config agree < 1e-12 per
        value; output dir carries config snapshot + version;
  M10-4 10 broken YAMLs -> located error messages, no bare traceback;
  M10-5 `lithofem selftest` fast tier green in a fresh virtualenv (separate
        full-tier test); GPU items auto-skip without a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from lithofem import config, driver
from lithofem.cli import run as cli_run

from .data_gen.gen_bad_yaml import BAD_CONFIGS

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.yaml"))
FAST_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "multilayer_diff.yaml"

needs_solver = pytest.mark.skipif(not driver.SOLVER_BIN.exists(),
                                  reason="lithofem_solve not built")


@pytest.mark.fast
@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_m10_4_bad_yaml_clean_errors(name: str, tmp_path: Path) -> None:
    cfg = tmp_path / f"{name}.yaml"
    cfg.write_text(BAD_CONFIGS[name])
    res = subprocess.run(
        [sys.executable, "-m", "lithofem.cli", "run", str(cfg),
         "-o", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode != 0
    assert "Traceback" not in res.stderr, res.stderr
    assert "configuration error" in res.stderr or "error:" in res.stderr


@pytest.mark.fast
def test_m10_2_dataclass_yaml_passes_schema(tmp_path: Path) -> None:
    import lithofem

    c = lithofem.Config(
        lx=48, ly=48, z_min=0, z_max=60, wavelength=13.5,
        materials={"absorber": {"n": 0.95, "k": 0.031}},
        layers=[lithofem.Layer(z=[0, 30], material="absorber")],
        frustums=[lithofem.Frustum(
            vertices=[[12, 12], [36, 12], [36, 36], [12, 36]], z0=0, h=30)],
        sources=[lithofem.PlaneWave(theta=6)],
        planes=[lithofem.OutputPlane(z=40, file="nf.h5")],
    )
    path = c.save(tmp_path / "built.yaml")
    model = config.load_yaml(path)  # must validate cleanly
    assert model.wavelength == 13.5
    assert len(model.frustums) == 1


@pytest.mark.full
@needs_solver
def test_m10_2_3_api_cli_identical_and_reproducible(tmp_path: Path) -> None:
    import lithofem

    out_cli = tmp_path / "cli"
    res = subprocess.run(
        [sys.executable, "-m", "lithofem.cli", "run", str(FAST_EXAMPLE),
         "-o", str(out_cli)],
        capture_output=True, text=True, timeout=3600,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    out_api = lithofem.run(FAST_EXAMPLE, tmp_path / "api")

    # M10-2: byte-identical HDF5 (timestamps disabled in the writer)
    fa = (out_cli / "g0_nearfield.h5").read_bytes()
    fb = (Path(out_api) / "g0_nearfield.h5").read_bytes()
    assert fa == fb

    # M10-3: numeric reproducibility across independent runs
    out2 = cli_run(FAST_EXAMPLE, tmp_path / "rep")
    with h5py.File(out_cli / "g0_nearfield.h5") as f1, \
         h5py.File(Path(out2) / "g0_nearfield.h5") as f2:
        for key in ("E_re", "E_im", "H_re", "H_im"):
            a, b = f1[key][:], f2[key][:]
            scale = max(np.abs(a).max(), 1.0)
            assert np.abs(a - b).max() < 1e-12 * scale, key
    # snapshot + version present
    assert (out_cli / "config_snapshot.yaml").exists()
    assert (out_cli / "version.json").exists()
    assert (out_cli / "run_log.jsonl").exists()


@pytest.mark.full
@needs_solver
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.stem for e in EXAMPLES])
def test_m10_1_examples_run(example: Path, tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, "-m", "lithofem.cli", "run", str(example),
         "-o", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=3600,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    outdir = tmp_path / "out"
    assert (outdir / "run_log.jsonl").exists()
    assert (outdir / "solve.json").exists()
    assert list(outdir.glob("g0_*.h5")), list(outdir.iterdir())


@pytest.mark.full
def test_m10_5_selftest_fast_in_fresh_venv(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                    str(venv)], check=True, timeout=600)
    py = venv / "bin" / "python"
    root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [str(py), "-m", "pip", "install", "-q", "-e", str(root),
         "--no-deps"], capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, res.stderr
    res = subprocess.run(
        [str(py), "-m", "lithofem.cli", "selftest"],
        capture_output=True, text=True, timeout=3600, cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert " passed" in res.stdout
