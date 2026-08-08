"""V2.5-M1 acceptance tests: assembly data extraction + CPU reassembly.

Wraps solver/bin/asm_test (see solver/asm_test.cpp) over the U5/U7 case
matrix of docs/gpu.md three geometries (multilayer difference
layer, line grating, 3D square hole) x p = 1..3 x theta in {0, 6} deg:

  U1: extracted ND reference tables == MFEM CalcVShape/CalcCurlShape
      (p = 1..4, every point of both default integration rules);
  U2: extracted affine geometry == MFEM element transformations
      (Jacobian, point map, total volume);
  U7: CPU reference reassembler (extraction data, GPU operation order)
      vs SesquilinearForm::Assemble(0): identical sparsity, values
      rel < 1e-13.

All cases are seconds-level small meshes (test economy, docs/gpu.md).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from lithofem import config, driver, meshgen

from . import test_solver_m6 as m6
from . import test_solver_m8 as m8

ASM_BIN = driver.SOLVER_BIN.parent / "asm_test"

pytestmark = [
    pytest.mark.full,
    pytest.mark.skipif(not ASM_BIN.exists(), reason="asm_test not built"),
]


def _case_config(name: str, order: int, theta: float) -> dict:
    if name == "multilayer":
        c = m6._multilayer_config(13.5, theta, "s", diff_layer=True)
        c["fem"] = {"order": order, "elems_per_wavelength": 3}
    elif name == "grating":
        # compact variant of the M8-1 line grating (assembly does not need
        # resolved physics; small-mesh test economy, docs/gpu.md)
        c = m8._grating_config(0.5, "TE")
        c["domain"] = {"Lx": 24.0, "Ly": 6.0, "z_min": -10.0, "z_max": 26.0}
        c["frustums"] = [{
            "vertices": [[0.0, 0.0], [12.0, 0.0], [12.0, 6.0], [0.0, 6.0]],
            "z0": 0.0, "h": 16.0, "alpha": 90, "epsilon": "absorber",
        }]
        c["output"]["planes"] = [
            {"z": 22.0, "quantities": ["E"], "resolution": [8, 4],
             "file": "top.h5"},
            {"z": -6.0, "quantities": ["E"], "resolution": [8, 4],
             "file": "bot.h5"},
        ]
        c["sources"][0]["incidence"]["theta"] = theta
        c["fem"] = {"order": order, "elems_per_wavelength": 4}
    else:  # hole3d
        c = m8._hole_config(m8.SQUARE, order)
        c["sources"][0]["incidence"]["theta"] = theta
        c["fem"] = {"order": order, "elems_per_wavelength": 3}
    return c


def _run_asm_bin(c: dict, tmp: Path, extra: list[str]) -> str:
    model = config.expand(c)
    prep = driver.prepare(model, tmp)
    per = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    res = subprocess.run(
        [str(ASM_BIN), "-m", str(per), "-j", str(prep.solve_json_path),
         "-g", "0", *extra],
        capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def _grab(out: str, keys: tuple[str, ...]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for key in keys:
        mt = re.search(rf"{key} ([\d.eE+-]+)", out)
        assert mt, f"missing {key} in asm_test output:\n{out}"
        vals[key] = float(mt.group(1))
    mt = re.search(r"kpar ([\d.eE+-]+) ([\d.eE+-]+)", out)
    vals["kpar"] = abs(float(mt.group(1))) + abs(float(mt.group(2)))
    return vals


def _run_asm_test(c: dict, tmp: Path) -> dict[str, float]:
    out = _run_asm_bin(c, tmp, [])
    return _grab(out, ("u1_max_abs", "u2_J_rel", "u2_x_abs", "u2_vol_rel",
                       "u7_struct_ok", "u7_rel", "ndof",
                       "timing_extract_s", "timing_csr_s",
                       "timing_mfem_assemble_s", "timing_reassemble_s"))


@pytest.mark.parametrize("theta", [0.0, 6.0], ids=["t0", "t6"])
@pytest.mark.parametrize("order", [1, 2, 3], ids=["p1", "p2", "p3"])
@pytest.mark.parametrize("case", ["multilayer", "grating", "hole3d"])
def test_u1_u2_u7_reassembly(case: str, order: int, theta: float,
                             tmp_path: Path) -> None:
    out = _run_asm_test(_case_config(case, order, theta), tmp_path)
    # theta=6 must actually exercise the kpar cross terms
    if theta > 0:
        assert out["kpar"] > 1e-6
    else:
        assert out["kpar"] == 0.0
    # U1: table extraction is the same arithmetic as MFEM's calls
    assert out["u1_max_abs"] == 0.0
    # U2: affine geometry (values in nm; x compare absolute)
    assert out["u2_J_rel"] < 1e-13
    assert out["u2_x_abs"] < 1e-10
    assert out["u2_vol_rel"] < 1e-12
    # U7: bitwise sparsity + element-wise values (hard 1e-13 criterion)
    assert out["u7_struct_ok"] == 1
    assert out["u7_rel"] < 1e-13


# ---- V2.5-M2: U3/U4 GPU local-matrix kernels ---------------------------

U3_KEYS = ("u3_curlcurl_rel", "u3_mass_rel")
U3_KPAR_KEYS = ("u3_cross0_rel", "u3_cross1_rel", "u3_klk_rel")


def _check_u3_u4(c: dict, order: int, theta: float, tmp: Path) -> None:
    out = _run_asm_bin(c, tmp, ["-p", str(order), "-skip-cpu", "-gpu"])
    keys = U3_KEYS + (U3_KPAR_KEYS if theta > 0 else ())
    vals = _grab(out, keys + ("u4_rel", "u4_elems", "fo_coverage"))
    for k in keys:
        assert vals[k] < 1e-13, (k, vals[k])
    # U4: full local matrix + dual DofTransformation over EVERY element
    assert vals["u4_rel"] < 1e-13, vals["u4_rel"]
    if order >= 2:
        # all face orientations that can occur in a conforming, consistently
        # oriented MFEM tet mesh: the shared face inherits Elem1's vertex
        # order (orientation 0), the other side sees a reflection (1/3/5);
        # pure rotations 2/4 are structurally impossible (verified on all
        # three case meshes; docs/gpu.md).
        assert vals["fo_coverage"] == 43, vals["fo_coverage"]


@pytest.mark.gpu
@pytest.mark.parametrize("theta", [0.0, 6.0], ids=["t0", "t6"])
@pytest.mark.parametrize("order", [1, 2, 3, 4], ids=["p1", "p2", "p3", "p4"])
def test_u3_u4_gpu_local_grating(order: int, theta: float,
                                 tmp_path: Path) -> None:
    # base mesh from the p1 config; -p overrides the ND order up to 4
    _check_u3_u4(_case_config("grating", 1, theta), order, theta, tmp_path)


@pytest.mark.gpu
@pytest.mark.parametrize("order", [2, 3], ids=["p2", "p3"])
@pytest.mark.parametrize("case", ["multilayer", "hole3d"])
def test_u3_u4_gpu_local_cases(case: str, order: int, tmp_path: Path) -> None:
    _check_u3_u4(_case_config(case, 1, 6.0), order, 6.0, tmp_path)


# ---- V2.5-M3: U5/U6 global assembly + elimination, U8/U9 e2e -----------

@pytest.mark.gpu
@pytest.mark.parametrize("theta", [0.0, 6.0], ids=["t0", "t6"])
@pytest.mark.parametrize("order", [1, 2, 3], ids=["p1", "p2", "p3"])
@pytest.mark.parametrize("case", ["multilayer", "grating", "hole3d"])
def test_u5_u6_gpu_global(case: str, order: int, theta: float,
                          tmp_path: Path) -> None:
    out = _run_asm_bin(_case_config(case, order, theta), tmp_path,
                       ["-skip-cpu", "-gpu-global"])
    vals = _grab(out, ("u5_struct_ok", "u5_rel", "u6_struct_ok", "u6_rel",
                       "u6_b_max"))
    assert vals["u5_struct_ok"] == 1
    assert vals["u5_rel"] < 1e-13
    assert vals["u6_struct_ok"] == 1
    assert vals["u6_rel"] < 1e-13
    assert vals["u6_b_max"] == 0.0


def _solve_raw(c: dict, tmp: Path, extra: list[str],
               env_extra: dict | None = None):
    import os
    model = config.expand(c)
    prep = driver.prepare(model, tmp)
    per = meshgen.mfem_periodic_mesh_path(prep.mesh_path)
    env = dict(os.environ)
    env.update(env_extra or {})
    res = subprocess.run(
        [str(driver.SOLVER_BIN), "-m", str(per), "-j",
         str(prep.solve_json_path), "-o", str(tmp), "-g", "0", *extra],
        capture_output=True, text=True, timeout=3600, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def _plane(tmp: Path, p: int):
    import numpy as np
    return np.fromfile(tmp / f"plane_g0_p{p}.bin")


@pytest.mark.gpu
@pytest.mark.cudss
def test_u8_e2e_gpu_vs_cpu_assembly(tmp_path: Path) -> None:
    import numpy as np
    c = _case_config("grating", 2, 6.0)
    c["solver"] = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
    c["fem"]["assembly"] = "gpu"
    gdir = tmp_path / "gpu"
    out = _solve_raw(c, gdir, [])
    # config passthrough engaged the zero-copy path; no test-only D2H
    assert "device-CSR zero-copy path" in out
    assert "asm_gpu_test_download" not in out
    assert "falling back" not in out
    c["fem"]["assembly"] = "cpu"
    cdir = tmp_path / "cpu"
    out_cpu = _solve_raw(c, cdir, [])
    assert "asm_gpu" not in out_cpu
    for p in (0, 1):
        a, b = _plane(gdir, p), _plane(cdir, p)
        rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
        assert rel < 1e-10, (p, rel)


@pytest.mark.gpu
@pytest.mark.cudss
def test_u8_cli_override_and_fallback(tmp_path: Path) -> None:
    import numpy as np
    c = _case_config("grating", 2, 6.0)
    c["solver"] = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
    # CLI -a gpu overrides the config default (cpu)
    gdir = tmp_path / "gpu"
    out = _solve_raw(c, gdir, ["-a", "gpu"])
    assert "device-CSR zero-copy path" in out
    # injected failure -> CPU assembly completes with an explicit log
    fdir = tmp_path / "fb"
    out_fb = _solve_raw(c, fdir, ["-a", "gpu"],
                        {"LITHOFEM_ASM_FORCE_FAIL": "1"})
    assert "forced failure injected" in out_fb
    assert "falling back to MFEM cpu assembly" in out_fb
    for p in (0, 1):
        a, b = _plane(gdir, p), _plane(fdir, p)
        rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
        assert rel < 1e-10, (p, rel)


# ---- V2.5-M4: e2e accuracy through the GPU-assembly path ----------------

@pytest.mark.full
@pytest.mark.gpu
@pytest.mark.cudss
def test_m4_tmm_gpu_assembly(tmp_path: Path) -> None:
    """M6-2 difference layer vs TMM analytic, gpu assembly + gpu solve."""
    import numpy as np
    c = m6._multilayer_config(13.5, 6.0, "s", diff_layer=True)
    c["fem"] = {"order": 3, "elems_per_wavelength": 10, "assembly": "gpu"}
    c["solver"] = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
    model = config.expand(c)
    prep = driver.prepare(model, tmp_path)
    meta = driver.solve_group(prep, 0)
    assert "device-CSR zero-copy path" in meta["stdout"]
    e_tot, _ = driver.total_field_on_plane(prep, 0, 0)
    fr = model.frustums[0]
    ref = m6._tmm_reference_field(model, (fr.geom.z_lo, fr.geom.z_hi, fr.eps))
    z = model.output.planes[0].z
    e_ref_hat = ref.eval(np.array([z]))[0]
    x, yv = driver.plane_grid(model, 0)
    kx, ky = model.groups[0].kpar
    phase = np.exp(1j * (ky * yv[:, None] + kx * x[None, :]))
    e_ref = e_ref_hat[None, None, :] * phase[..., None]
    rel = np.linalg.norm(e_tot - e_ref) / np.linalg.norm(e_ref)
    print(f"m4_tmm_rel {rel:.3e}")
    # discretization-error level, unchanged from v1/v2 (~8e-5 at epw 10)
    assert rel < 1e-4, rel


@pytest.mark.full
@pytest.mark.gpu
@pytest.mark.cudss
def test_m4_orders_gpu_vs_cpu_assembly(tmp_path: Path) -> None:
    """M8-1 line grating: order efficiencies, gpu vs cpu assembly < 1e-8."""
    c = m8._grating_config(0.5, "TE")
    c["solver"] = {"type": "direct", "device": "gpu", "gpu_ids": [0]}
    c["fem"] = dict(c["fem"], assembly="gpu")
    rg, tg = m8._fem_efficiencies(c, tmp_path / "g")
    c["fem"] = dict(c["fem"], assembly="cpu")
    rc_, tc_ = m8._fem_efficiencies(c, tmp_path / "c")
    worst = 0.0
    for a, b in ((rg, rc_), (tg, tc_)):
        for k in set(a) | set(b):
            worst = max(worst, abs(a.get(k, 0.0) - b.get(k, 0.0)))
    print(f"m4_orders_max_abs {worst:.3e}")
    assert worst < 1e-8, worst


def test_u9_default_assembly_unchanged(tmp_path: Path) -> None:
    # fem.assembly defaults to cpu; the default path never touches the new
    # code (no asm_gpu lines) and solve.json carries the key
    c = _case_config("multilayer", 1, 0.0)
    model = config.expand(c)
    assert model.fem.assembly == "cpu"
    out = _solve_raw(c, tmp_path, [])
    assert "asm_gpu" not in out
    assert "residual" in out
