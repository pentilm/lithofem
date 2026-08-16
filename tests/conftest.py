"""Shared test configuration.

Two things happen here beyond plain fixtures:

1. GPU/cuDSS-marked tests auto-skip when the hardware or library is absent, so
   the suite stays meaningful on a CPU-only machine.
2. Modules marked `gpu_ok` - the expensive physics validation - run on the
   GPU when one is available (`--device=auto`, the default). This changes
   *where* the physics is computed, not *what* is checked: the two paths are
   pinned to each other by test_asm_v25.py (<1e-13 element-wise) and
   test_solver_v2.py (<5e-13 in the fields), both of which run the CPU path
   explicitly. `--device=cpu` disables the routing.
3. GPU tests hold a shared lock and `multigpu` tests an exclusive one, so a
   parallel run cannot starve them of VRAM or of the second card.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=20)
        if out.returncode != 0:
            return 0
        return sum(1 for ln in out.stdout.splitlines() if ln.startswith("GPU "))
    except (OSError, subprocess.TimeoutExpired):
        return 0


GPU_COUNT = _gpu_count()
HAS_GPU = GPU_COUNT > 0


def _has_cudss() -> bool:
    """cuDSS runtime present (pip wheel `nvidia-cudss-cu13`, design note)?"""
    if not HAS_GPU:
        return False
    import ctypes
    import glob

    candidates = glob.glob(
        "/opt/conda/lib/python3*/site-packages/nvidia/cu13/lib/libcudss.so*"
    ) + glob.glob("/usr/local/cuda*/lib64/libcudss.so*")
    for path in candidates:
        try:
            ctypes.CDLL(path)
            return True
        except OSError:
            continue
    return False


HAS_CUDSS = _has_cudss()


def pytest_addoption(parser) -> None:  # noqa: ANN001
    parser.addoption(
        "--device", action="store", default="auto",
        choices=("auto", "gpu", "cpu"),
        help="where solver tests that express no preference run: auto (GPU "
             "when available, the default), gpu (require it), cpu (never "
             "redirect)",
    )


def pytest_configure(config) -> None:  # noqa: ANN001
    """Divide the cores among xdist workers instead of oversubscribing them.

    Each test spawns a solver subprocess that is itself threaded (OpenBLAS
    under UMFPACK), and meshing runs threaded inside the worker. Without a cap
    every worker behaves as if it owned the machine: N workers x M threads
    thrashes the scheduler, and OpenBLAS with too many threads has been
    observed to fail outright on large factorizations (observed on large factorizations).

    An explicit setting in the environment always wins.
    """
    # in a worker xdist reports the count through the environment; in the
    # controller it is the -n option
    n = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 0)) or \
        (getattr(config.option, "numprocesses", None) or 1)
    if n <= 1:
        return
    share = max(1, (os.cpu_count() or 1) // n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(min(share, 32)))
    # deliberately NOT LITHOFEM_MESH_THREADS: threaded meshing is not
    # reproducible, and a dozen tests compare two runs of the same
    # configuration (GPU vs CPU, API vs CLI, iterative vs direct). Give them
    # different meshes and they compare different problems - which is exactly
    # what happened when this cap first included it.


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU detected (auto-skip)")
    skip_cudss = pytest.mark.skip(reason="cuDSS not available (auto-skip)")
    for item in items:
        if not HAS_GPU and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not HAS_CUDSS and "cudss" in item.keywords:
            item.add_marker(skip_cudss)


def _worker_gpu() -> int:
    """Which card this xdist worker uses (round-robin over the cards).

    Spreading workers across cards keeps a parallel run from piling every
    factorization onto card 0 and tripping the VRAM guard.
    """
    if GPU_COUNT <= 1:
        return 0
    wid = os.environ.get("PYTEST_XDIST_WORKER", "")   # 'gw0', 'gw1', ...
    try:
        return int(wid[2:]) % GPU_COUNT
    except ValueError:
        return 0


@pytest.fixture(autouse=True)
def _solver_device(request, monkeypatch):  # noqa: ANN001
    """Route the expensive physics-validation tests onto the GPU.

    Opt-in by design. An earlier version redirected every configuration that
    did not name a solver, which silently changed the defaults that a dozen
    tests exist to assert (config defaults, CLI/API equivalence, the UMFPACK
    memory-limit path). Whether a test may move is a property of the test, so
    the test says so: `gpu_ok` on the module or the function.

    Moving them is legitimate because GPU and CPU are separately pinned to
    each other (element-wise <1e-13 in test_asm_v25, fields <5e-13 in
    test_solver_v2, both of which run the CPU path explicitly), so this
    changes where the physics is computed, not what is checked.
    """
    mode = request.config.getoption("--device")
    # cpu_reference wins over a module-level gpu_ok: those tests exist to
    # exercise the CPU path itself
    if (mode == "cpu" or "gpu_ok" not in request.keywords
            or "cpu_reference" in request.keywords):
        return
    if not (HAS_GPU and HAS_CUDSS) and mode != "gpu":
        return

    from lithofem import config as cfg

    gpu_id = _worker_gpu()
    orig_expand = cfg.expand

    def expand(raw):  # noqa: ANN001, ANN202
        if isinstance(raw, dict) and not raw.get("solver"):
            raw = dict(raw)
            raw["solver"] = {"type": "direct", "device": "gpu",
                             "gpu_ids": [gpu_id]}
            fem = dict(raw.get("fem") or {})
            if "assembly" not in fem:
                fem["assembly"] = "gpu"
                raw["fem"] = fem
        return orig_expand(raw)

    monkeypatch.setattr(cfg, "expand", expand)


def _gpu_slots(gpu_id: int) -> int:
    """How many GPU tests may share one card concurrently.

    The biggest single test needs ~17 GB (M8-1 at p=3); budget 24 GB per
    slot so the sum of concurrent factorizations stays inside the card even
    if several land near the maximum. On a 141 GB H200 that is 5 concurrent
    tests per card; on a 24 GB desktop card it degrades to 1, i.e. GPU tests
    simply serialize instead of tripping the fallback that several of them
    assert never happens.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=20)
        total_gb = float(out.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 1
    return max(1, int(total_gb // 24))


@pytest.fixture(autouse=True)
def _gpu_arbitration(request):  # noqa: ANN001
    """Bound and serialize GPU usage across xdist workers.

    Two mechanisms, expressing two different constraints:

    * a reader/writer lock: `multigpu` tests (the R2-10 sweep test asserts
      both cards are bound, that nothing fell back, and that parallel beat
      serial) take it exclusively; every other GPU test holds it shared, so
      the multi-GPU test gets the whole machine to itself;
    * per-card counting slots: shared holders are *bounded*, not unlimited -
      N slot files per card, each a try-lock in round-robin, so concurrent
      VRAM use stays inside the card. Without this, eleven workers can pile
      onto one card and exhaust it, which is exactly what the first full run
      did (the equivalence tests then "failed" because cuDSS fell back to
      the CPU mid-comparison).
    """
    if not HAS_GPU:
        yield
        return
    exclusive = "multigpu" in request.keywords
    uses_gpu = exclusive or "gpu" in request.keywords or \
        "gpu_ok" in request.keywords
    if not uses_gpu:
        yield
        return

    tmp = Path(tempfile.gettempdir())
    main_lock = open(tmp / "lithofem-gpu.lock", "a+")
    fcntl.flock(main_lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    slot = None
    try:
        if not exclusive:
            gpu_id = _worker_gpu()
            n_slots = _gpu_slots(gpu_id)
            # block on a fixed slot derived from the worker id: with workers
            # round-robined over cards this spreads them over slots too, and
            # blocking (rather than spinning over try-locks) keeps it simple
            wid = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
            idx = (int(wid[2:]) if wid[2:].isdigit() else 0) // max(GPU_COUNT, 1)
            slot = open(tmp / f"lithofem-gpu{gpu_id}-slot{idx % n_slots}.lock",
                        "a+")
            fcntl.flock(slot, fcntl.LOCK_EX)
        yield
    finally:
        if slot is not None:
            fcntl.flock(slot, fcntl.LOCK_UN)
            slot.close()
        fcntl.flock(main_lock, fcntl.LOCK_UN)
        main_lock.close()
